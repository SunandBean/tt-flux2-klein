# SPDX-License-Identifier: Apache-2.0
"""FLUX.2 [klein] 4B text-to-image and reference editing on one P100a: TT text encoder + DiT + VAE decoder,
host scheduler / reference VAE encode."""
from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from typing import List, Optional

import torch

from . import host
from .config import SCHED, snapshot_dir, text_encoder_dir

# device helpers are model-independent
from .device import Timing, close_device, dram_stats, open_device  # noqa: F401


class KleinTT:
    def __init__(self, dev, snapshot: Optional[str] = None, te_dir: Optional[str] = None, dit_prec=None, te_prec=None,
                 device_vae: bool = True):
        from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
        from transformers import AutoTokenizer
        import ttnn

        from .tt_dit import KleinDiT
        from .tt_text_encoder import KleinTextEncoder
        from .tt_vae import KleinVAE

        self.dev = dev
        self.snapshot = snapshot or snapshot_dir()
        te_dir = te_dir or text_encoder_dir()
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.snapshot, "tokenizer"))
        self.te = KleinTextEncoder(dev, host.LazyCheckpoint(os.path.dirname(te_dir), os.path.basename(te_dir)), te_prec)
        self.dit = KleinDiT(dev, host.LazyCheckpoint(self.snapshot, "transformer"), dit_prec)
        self.vae = KleinVAE(dev, self.snapshot, device_decode=device_vae)
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=16)
        ttnn.synchronize_device(dev)
        self.load_s = time.perf_counter() - t0
        self._prep = None
        self._ref_cache: "OrderedDict[str, tuple]" = OrderedDict()  # continuous runs reuse one reference

    def prepared(self, geo: host.Geometry):
        if self._prep is None or self._prep.geo != geo:
            if self._prep is not None:
                self.dit.release(self._prep)
                if (self._prep.geo.width, self._prep.geo.height) != (geo.width, geo.height):
                    self.dev.clear_program_cache()  # conv2d L1_SMALL configs accumulate per size (Z-Image diag_vae)
            self._prep = self.dit.prepare(geo)
        return self._prep

    def encode(self, prompt: str) -> torch.Tensor:
        ids, valid = host.tokenize(self.tokenizer, prompt)
        return self.te.encode(ids, valid)

    REF_CACHE_SIZE = 4

    def references(self, images: List) -> tuple:
        tokens, grids = [], []
        for img in images:
            rgb = img.convert("RGB")
            key = hashlib.sha256(repr(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if key in self._ref_cache:
                self._ref_cache.move_to_end(key)
                tok, grid = self._ref_cache[key]
            else:
                tensor, grid = host.preprocess_reference(rgb, self.image_processor)
                tok = self.vae.encode_reference(tensor)
                self._ref_cache[key] = (tok, grid)
                while len(self._ref_cache) > self.REF_CACHE_SIZE:
                    self._ref_cache.popitem(last=False)
            tokens.append(tok)
            grids.append(grid)
        return (torch.cat(tokens) if tokens else None), tuple(grids)

    def denoise(self, prompt_embeds: torch.Tensor, width: int, height: int, seed: int, timing: Timing,
                ref_tokens: Optional[torch.Tensor] = None, ref_grids: tuple = (), steps: int = SCHED.steps):
        import ttnn

        geo = host.Geometry(width=width, height=height, refs=ref_grids)
        t0 = time.perf_counter()
        prep = self.prepared(geo)
        ctx = self.dit.context(prompt_embeds)
        timing.mark("context_s", t0)
        sched = host.make_scheduler(self.snapshot, geo, steps)
        latents = host.init_noise(geo, seed)[None]  # [1, N, 128] bf16 like the pipeline
        for t in sched.timesteps:
            t0 = time.perf_counter()
            cond = self.dit.condition(float(t))
            out = self.dit.step(latents[0], ref_tokens, ctx, cond, prep)
            self.dit.free_condition(cond)
            timing.mark("dit_s", t0)
            latents = sched.step(out.to(torch.bfloat16)[None], t, latents, return_dict=False)[0]
        ttnn.deallocate(ctx)
        return self.vae.stats.denormalize(latents[0], geo), geo

    def decode(self, vae_latents: torch.Tensor):
        image = self.vae.decode(vae_latents)
        return self.image_processor.postprocess(image, output_type="pil")[0], image

    def generate(self, prompt: str, width: int, height: int, seed: int, images: Optional[List] = None):
        timing = Timing()
        t0 = time.perf_counter()
        emb = self.encode(prompt)
        timing.mark("text_encoder_s", t0)
        ref_tokens, grids = None, ()
        if images:
            t0 = time.perf_counter()
            ref_tokens, grids = self.references(images)
            timing.mark("reference_encode_s", t0)
        lat, _ = self.denoise(emb, width, height, seed, timing, ref_tokens, grids)
        t0 = time.perf_counter()
        image, _ = self.decode(lat)
        timing.mark("vae_decode_s", t0)
        return image, timing

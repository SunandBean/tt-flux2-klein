# SPDX-License-Identifier: Apache-2.0
"""Host-side (CPU) parts of the single-P100a FLUX.2 [klein] 4B port.

Mirrors diffusers 0.37.1 Flux2KleinPipeline / Flux2Transformer2DModel for batch 1 (checked in
experiments/flux2-klein/tests/test_host.py):
  * prompt: chat template (enable_thinking=False), padded to 512 tokens; every row is kept and attended
  * joint token layout used on the device: [text (512) ; image (N) ; references (R) ; pad] with pad keys
    masked, RoPE ids (t, h, w, l) per token, cos/sin in the adjacent-pair layout of rotary_embedding_llama
  * latents: (1, 128, H/16, W/16) bf16 noise, row-major packing; BN de-normalization + unpatchify at the end
  * schedule: sigmas linspace(1, 1/N, N) with the empirical dynamic-shift mu of the image token count
  * timestep conditioning evaluated in bf16 exactly like the reference; LayerNorm modulation is folded
    into (weight = 1 + scale, bias = shift)
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import DIT, REF_T_SCALE, SCHED, TE, TILE, DiTConfig

# shared, model-independent helpers (Apache-2.0; see common.py, from the tt-z-image-turbo port)
from .common import (LazyCheckpoint, fuse_qkv, interleave_pairs_permutation, linear_to_mm, pad_rows,  # noqa: F401
                     permute_heads_rows, rot_transformation_mat, round_up, swiglu_interleave)

MASK_NEG = -1e9  # additive bias for masked keys (bf16-representable, same as the Qwen-Image-2.1 port)


# ----------------------------------------------------------------------------- geometry
@dataclass(frozen=True)
class Geometry:
    width: int
    height: int
    refs: Tuple[Tuple[int, int], ...] = ()  # packed (grid_h, grid_w) of each reference image

    @property
    def grid_h(self) -> int:
        return self.height // 16

    @property
    def grid_w(self) -> int:
        return self.width // 16

    @property
    def n_img(self) -> int:
        return self.grid_h * self.grid_w

    @property
    def n_ref(self) -> int:
        return sum(h * w for h, w in self.refs)

    @property
    def n_spatial(self) -> int:
        return self.n_img + self.n_ref

    @property
    def n_spatial_pad(self) -> int:
        return round_up(self.n_spatial, TILE)

    @property
    def n_txt(self) -> int:
        return TE.seq_len

    @property
    def n_joint(self) -> int:
        return self.n_txt + self.n_spatial_pad

    @property
    def n_pad(self) -> int:
        return self.n_spatial_pad - self.n_spatial


def text_ids(n: int = TE.seq_len) -> torch.Tensor:
    return torch.stack([torch.zeros(n, dtype=torch.long)] * 3 + [torch.arange(n)], dim=-1)


def image_ids(grid_h: int, grid_w: int, t: int = 0) -> torch.Tensor:
    hh = torch.arange(grid_h).repeat_interleave(grid_w)
    ww = torch.arange(grid_w).repeat(grid_h)
    return torch.stack([torch.full_like(hh, t), hh, ww, torch.zeros_like(hh)], dim=-1)


def joint_ids(geo: Geometry) -> torch.Tensor:
    """[n_joint, 4] ids in device order: text, image, references (t = 10 + 10 i), pad (ids 0; keys masked)."""
    parts = [text_ids(geo.n_txt), image_ids(geo.grid_h, geo.grid_w)]
    for i, (h, w) in enumerate(geo.refs):
        parts.append(image_ids(h, w, REF_T_SCALE + REF_T_SCALE * i))
    parts.append(torch.zeros(geo.n_pad, 4, dtype=torch.long))
    return torch.cat(parts)


def cos_sin(ids: torch.Tensor, cfg: DiTConfig = DIT, dtype=torch.bfloat16):
    """Flux2PosEmbed: per axis get_1d_rotary_pos_embed(32, pos, theta=2000, repeat_interleave_real=True)."""
    cos, sin = [], []
    for i, d in enumerate(cfg.axes_dims):
        freqs = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
        ang = torch.outer(ids[:, i].to(torch.float64), freqs)
        cos.append(ang.cos().float().repeat_interleave(2, -1))
        sin.append(ang.sin().float().repeat_interleave(2, -1))
    return torch.cat(cos, -1).to(dtype), torch.cat(sin, -1).to(dtype)


def key_mask(geo: Geometry, dtype=torch.bfloat16):
    """Additive [1, 1, S, S] bias masking the pad keys, or None when the spatial tokens fill whole tiles."""
    if geo.n_pad == 0:
        return None
    m = torch.zeros(geo.n_joint, dtype=torch.float32)
    m[geo.n_txt + geo.n_spatial:] = MASK_NEG
    return m.expand(geo.n_joint, geo.n_joint).reshape(1, 1, geo.n_joint, geo.n_joint).to(dtype)


def apply_rope_adjacent(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
    return x * cos + rot * sin


# ----------------------------------------------------------------------------- latents
def init_noise(geo: Geometry, seed: int) -> torch.Tensor:
    """prepare_latents: randn (1, 128, H/16, W/16) directly in bf16 from a CPU generator, packed [N, 128]."""
    g = torch.Generator("cpu").manual_seed(int(seed))
    lat = torch.randn((1, DIT.in_channels, geo.grid_h, geo.grid_w), generator=g, dtype=torch.bfloat16)
    return pack(lat)


def pack(lat: torch.Tensor) -> torch.Tensor:
    """(1, C, h, w) -> (h * w, C), row-major like _pack_latents."""
    return lat.reshape(lat.shape[1], -1).t().contiguous()


def unpack(x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    return x[: grid_h * grid_w].t().reshape(1, -1, grid_h, grid_w)


def patchify(lat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = lat.shape
    return lat.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def unpatchify(lat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = lat.shape
    return lat.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)


@dataclass
class LatentStats:
    """VAE BatchNorm running stats used to (de)normalize patchified latents."""

    mean: torch.Tensor  # [1, 128, 1, 1]
    std: torch.Tensor

    @classmethod
    def from_vae(cls, vae) -> "LatentStats":
        mean = vae.bn.running_mean.view(1, -1, 1, 1)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
        return cls(mean, std)

    def denormalize(self, packed: torch.Tensor, geo: Geometry) -> torch.Tensor:
        """Final packed tokens -> (1, 32, H/8, W/8) VAE latents (dtype of the tokens, like the pipeline)."""
        lat = unpack(packed, geo.grid_h, geo.grid_w)
        lat = lat * self.std.to(lat.dtype) + self.mean.to(lat.dtype)
        return unpatchify(lat)

    def normalize(self, vae_latents: torch.Tensor) -> torch.Tensor:
        lat = patchify(vae_latents)
        return (lat - self.mean.to(lat.dtype)) / self.std.to(lat.dtype)


# ----------------------------------------------------------------------------- schedule
def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def make_scheduler(snapshot: str, geo: Geometry, steps: int = SCHED.steps):
    from diffusers import FlowMatchEulerDiscreteScheduler

    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(snapshot, subfolder="scheduler")
    sigmas = np.linspace(1.0, 1 / steps, steps)
    sched.set_timesteps(sigmas=sigmas, mu=compute_empirical_mu(geo.n_img, steps))
    sched.set_begin_index(0)
    return sched


# ----------------------------------------------------------------------------- conditioning
@dataclass
class StepCond:
    """bf16 [dim] rows of one timestep: LayerNorm (weight, bias) pairs and gates."""

    img: Dict[str, torch.Tensor]  # ln1_w, ln1_b, gate1, ln2_w, ln2_b, gate2
    txt: Dict[str, torch.Tensor]
    single: Dict[str, torch.Tensor]  # ln_w, ln_b, gate
    out_w: torch.Tensor
    out_b: torch.Tensor


class TimeConditioning:
    def __init__(self, ckpt, cfg: DiTConfig = DIT):
        self.cfg = cfg
        g = lambda k: ckpt.get(k, torch.bfloat16)
        self.l1 = g("time_guidance_embed.timestep_embedder.linear_1.weight")
        self.l2 = g("time_guidance_embed.timestep_embedder.linear_2.weight")
        self.mod_img = g("double_stream_modulation_img.linear.weight")
        self.mod_txt = g("double_stream_modulation_txt.linear.weight")
        self.mod_single = g("single_stream_modulation.linear.weight")
        self.norm_out = g("norm_out.linear.weight")

    def temb(self, t_sched: float) -> torch.Tensor:
        # pipeline: t.expand(B).to(bf16) / 1000 ; transformer: timestep.to(bf16) * 1000
        t = torch.tensor([t_sched], dtype=torch.float32).to(torch.bfloat16) / 1000
        t = t.to(torch.bfloat16) * 1000
        half = self.cfg.timestep_channels // 2
        exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        emb = t[:, None].float() * torch.exp(exponent)[None]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1).to(torch.bfloat16)  # flip_sin_to_cos
        return F.linear(F.silu(F.linear(emb, self.l1)), self.l2)

    def step(self, t_sched: float) -> StepCond:
        temb = self.temb(t_sched)
        act = F.silu(temb)
        d = self.cfg.dim
        one = torch.ones((), dtype=torch.bfloat16)

        def pairs(w, sets):
            chunks = F.linear(act, w)[0].split(d)
            out = {}
            for s in range(sets):
                shift, scale, gate = chunks[3 * s: 3 * s + 3]
                sfx = "" if sets == 1 else str(s + 1)
                out[f"ln{sfx}_w"], out[f"ln{sfx}_b"], out[f"gate{sfx}"] = one + scale, shift, gate
            return out

        scale, shift = F.linear(act, self.norm_out)[0].split(d)
        return StepCond(pairs(self.mod_img, 2), pairs(self.mod_txt, 2), pairs(self.mod_single, 1), one + scale, shift)


# ----------------------------------------------------------------------------- prompt
def tokenize(tokenizer, prompt: str):
    """Token ids [512] and attention mask [512] exactly as _get_qwen3_prompt_embeds builds them."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=TE.seq_len)
    return inputs["input_ids"][0], inputs["attention_mask"][0].bool()


def te_mask(valid: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """Additive [1, 1, L, L] bias: causal and padding keys masked (transformers' causal + padding mask)."""
    L = valid.shape[0]
    q = torch.arange(L)[:, None]
    k = torch.arange(L)[None, :]
    allowed = (k <= q) & valid[None, :]
    return torch.where(allowed, 0.0, MASK_NEG).reshape(1, 1, L, L).to(dtype)


def te_cos_sin(seq_len: int, dtype=torch.bfloat16):
    freqs = 1.0 / (TE.rope_theta ** (torch.arange(0, TE.head_dim, 2, dtype=torch.float64) / TE.head_dim))
    ang = torch.outer(torch.arange(seq_len, dtype=torch.float64), freqs).float()
    return ang.cos().repeat_interleave(2, -1).to(dtype), ang.sin().repeat_interleave(2, -1).to(dtype)


# ----------------------------------------------------------------------------- references
def preprocess_reference(image, image_processor, vae_scale_factor: int = 8):
    """Flux2KleinPipeline step 4: cap the area at 1024^2, crop to a multiple of 16, normalize to [-1, 1]."""
    image_processor.check_image_input(image)
    w, h = image.size
    if w * h > 1024 * 1024:
        image = image_processor._resize_to_target_area(image, 1024 * 1024)
        w, h = image.size
    m = vae_scale_factor * 2
    w, h = (w // m) * m, (h // m) * m
    return image_processor.preprocess(image, height=h, width=w, resize_mode="crop"), (h // 16, w // 16)

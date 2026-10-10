# SPDX-License-Identifier: Apache-2.0
"""FLUX.2 VAE (AutoencoderKLFlux2, 32 latent channels) decode on one Blackhole; encode stays on host.

The decoder is diffusers' generic Decoder with 32 input channels, so the tt_dit SD3.5/FLUX VAE decoder already
vendored for Z-Image (vendor_tt_dit/, with the non-square slicing / GroupNorm grid fixes) is
built from it directly. post_quant_conv (1x1, 32 -> 32) runs on host. Reference images are VAE-encoded on host
(argmax / distribution mode, like the pipeline) in bf16; moving the encoder to the device is a later optimization."""
from __future__ import annotations

import torch
import ttnn

from .vendor_tt_dit.models.vae.vae_sd35 import VAEDecoder
from .vendor_tt_dit.parallel.config import VAEParallelConfig
from .vendor_tt_dit.parallel.manager import CCLManager
from .vendor_tt_dit.utils import tensor as tt_tensor

from . import host


class KleinVAE:
    def __init__(self, dev, snapshot: str, device_decode: bool = True):
        from diffusers import AutoencoderKLFlux2

        self.torch_vae = AutoencoderKLFlux2.from_pretrained(snapshot, subfolder="vae", torch_dtype=torch.float32).eval()
        self.stats = host.LatentStats.from_vae(self.torch_vae)
        # The pipeline encodes references in bf16; on host that is also 2x faster than fp32
        # (check_ref_encode.py: 1024^2 10.9 s -> 5.8 s on 8 threads, token PCC vs golden 0.99993 -> 0.99998).
        # Only the encoder path is cast; post_quant_conv / decoder weights stay fp32.
        self.torch_vae.encoder.to(torch.bfloat16)
        if getattr(self.torch_vae, "quant_conv", None) is not None:
            self.torch_vae.quant_conv.to(torch.bfloat16)
        self.dev = dev
        self.decoder = None
        if device_decode:
            self.ccl = CCLManager(dev, num_links=1, topology=ttnn.Topology.Linear)
            self.decoder = VAEDecoder.from_torch(self.torch_vae.decoder, mesh_device=dev,
                                                 parallel_config=VAEParallelConfig.from_tuple((1, 1)),
                                                 ccl_manager=self.ccl)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """[1, 32, h, w] VAE latents -> [1, 3, 8h, 8w] float image in [-1, 1]."""
        z = self.torch_vae.post_quant_conv(latents.float())
        if self.decoder is None:
            return self.torch_vae.decoder(z)
        tt_in = tt_tensor.from_torch(z.permute(0, 2, 3, 1), device=self.dev)
        tt_out = self.decoder.forward(tt_in)
        out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2).float()
        ttnn.deallocate(tt_out)
        ttnn.deallocate(tt_in)
        return out

    @torch.no_grad()
    def encode_reference(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocessed [1, 3, H, W] image -> BN-normalized packed reference tokens [H/16 * W/16, 128] bf16."""
        dist = self.torch_vae.encode(image_tensor.to(torch.bfloat16)).latent_dist
        return host.pack(self.stats.normalize(dist.mode().float())).to(torch.bfloat16)

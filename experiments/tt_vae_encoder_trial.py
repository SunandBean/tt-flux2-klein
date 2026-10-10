# SPDX-License-Identifier: Apache-2.0
"""TRIAL, NOT USED: FLUX.2 VAE encoder (diffusers Encoder, 32 latent channels, double_z) on one Blackhole.

Measured 2026-09-26 (diag_vae_encoder.py): token PCC vs the CPU bf16 encoder only 0.52-0.70 (bug not located,
downsample path suspected); 1024x1024 fails (stride-2 conv needs <= 16 width slices, the fallback asked 32);
a reference shape seen for the first time costs 12-32 s of compile/weight preparation versus 1.3-5.7 s on
CPU, and repeated references are already served from the token cache. Production keeps the CPU bf16 encode.

Built from the vendored tt_dit VAE blocks (deploy/zimage/vendor_tt_dit: ResnetBlock, UnetMidBlock2D, Conv2d,
GroupNorm) that already run the decoder. The only new piece is diffusers' Downsample2D(padding=0): zero pad one
column right and one row bottom, then a stride-2 3x3 convolution. quant_conv (1x1) and the Gaussian mode
(first 32 channels) are applied on host by the caller."""
from __future__ import annotations

import ttnn

from zimage.vendor_tt_dit.layers.conv2d import Conv2d
from zimage.vendor_tt_dit.layers.module import Module, ModuleList
from zimage.vendor_tt_dit.layers.normalization import GroupNorm
from zimage.vendor_tt_dit.models.vae.vae_sd35 import ResnetBlock, UnetMidBlock2D


class Downsample2D(Module):
    def __init__(self, channels, *, mesh_device, parallel_config, ccl_manager):
        super().__init__()
        self.conv = Conv2d(channels, channels, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), mesh_device=mesh_device,
                           out_mesh_axis=parallel_config.tensor_parallel.mesh_axis, ccl_manager=ccl_manager,
                           use_barrier=False)

    def forward(self, x):
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)], 0.0)  # F.pad(x, (0, 1, 0, 1)) in NHWC
        return self.conv(x)


class DownEncoderBlock2D(Module):
    def __init__(self, *, in_channels, out_channels, num_layers, groups, add_downsample, mesh_device, parallel_config,
                 ccl_manager):
        super().__init__()
        self.resnets = ModuleList(
            ResnetBlock(in_channels=in_channels if i == 0 else out_channels, out_channels=out_channels, num_groups=groups,
                        eps=1e-6, mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager)
            for i in range(num_layers))
        self.downsamplers = ModuleList(
            [Downsample2D(out_channels, mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager)]
            if add_downsample else [])

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        for down in self.downsamplers:
            x = down(x)
        return x


class VAEEncoder(Module):
    def __init__(self, *, block_out_channels, in_channels, out_channels, layers_per_block, norm_num_groups, mesh_device,
                 parallel_config, ccl_manager):
        super().__init__()
        axis = parallel_config.tensor_parallel.mesh_axis
        self.conv_in = Conv2d(in_channels, block_out_channels[0], kernel_size=(3, 3), padding=(1, 1),
                              mesh_device=mesh_device, out_mesh_axis=axis, ccl_manager=ccl_manager, use_barrier=False)
        self.down_blocks = ModuleList()
        prev = block_out_channels[0]
        for i, ch in enumerate(block_out_channels):
            self.down_blocks.append(DownEncoderBlock2D(
                in_channels=prev, out_channels=ch, num_layers=layers_per_block, groups=norm_num_groups,
                add_downsample=i != len(block_out_channels) - 1, mesh_device=mesh_device,
                parallel_config=parallel_config, ccl_manager=ccl_manager))
            prev = ch
        self.mid_block = UnetMidBlock2D(in_channels=prev, resnet_groups=norm_num_groups, attention_head_dim=prev,
                                        mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager)
        self.conv_norm_out = GroupNorm(num_groups=norm_num_groups, num_channels=prev, eps=1e-6, mesh_device=mesh_device,
                                       mesh_axis=axis)
        self.conv_out = Conv2d(prev, out_channels, kernel_size=(3, 3), padding=(1, 1), mesh_device=mesh_device,
                               ccl_manager=ccl_manager)

    @classmethod
    def from_torch(cls, torch_ref, *, mesh_device, parallel_config, ccl_manager):
        model = cls(block_out_channels=[b.resnets[0].conv2.out_channels for b in torch_ref.down_blocks],
                    in_channels=torch_ref.conv_in.in_channels, out_channels=torch_ref.conv_out.out_channels,
                    layers_per_block=len(torch_ref.down_blocks[0].resnets),
                    norm_num_groups=torch_ref.mid_block.resnets[0].norm1.num_groups, mesh_device=mesh_device,
                    parallel_config=parallel_config, ccl_manager=ccl_manager)
        model.load_torch_state_dict(torch_ref.state_dict())
        return model

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        x = self.conv_norm_out(x)
        x = ttnn.silu(x)
        return self.conv_out(x)

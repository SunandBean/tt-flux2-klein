# SPDX-License-Identifier: Apache-2.0
"""Static configuration of black-forest-labs/FLUX.2-klein-4B @ e7b7dc2 (transformer/text_encoder/scheduler)."""
from __future__ import annotations

import os
from dataclasses import dataclass

HF_REPO = "black-forest-labs/FLUX.2-klein-4B"
HF_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
TILE = 32


@dataclass(frozen=True)
class DiTConfig:
    dim: int = 3072  # 24 heads x 128
    heads: int = 24
    head_dim: int = 128
    mlp_hidden: int = 9216  # int(dim * 3.0)
    n_double: int = 5
    n_single: int = 20
    in_channels: int = 128  # 32 VAE channels x 2x2 patch
    joint_attention_dim: int = 7680  # 3 Qwen3 layers x 2560
    axes_dims: tuple = (32, 32, 32, 32)  # (t, h, w, l)
    rope_theta: float = 2000.0
    eps: float = 1e-6  # block LayerNorms and qk RMSNorms
    timestep_channels: int = 256


@dataclass(frozen=True)
class TextEncoderConfig:
    num_layers: int = 27  # hidden_states[9], [18], [27] are used; layers 28..36 never run
    taps: tuple = (9, 18, 27)  # hidden_states index k == residual stream after k decoder layers
    hidden: int = 2560
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    intermediate: int = 9728
    rms_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    seq_len: int = 512  # Flux2KleinPipeline pads every prompt to max_sequence_length and keeps all 512 rows


@dataclass(frozen=True)
class SchedulerConfig:
    steps: int = 4  # model card: num_inference_steps=4, guidance_scale=1.0 (distilled, no CFG)


DIT = DiTConfig()
TE = TextEncoderConfig()
SCHED = SchedulerConfig()
REF_T_SCALE = 10  # reference image i gets t coordinate 10 + 10 * i


def snapshot_dir() -> str:
    env = os.environ.get("KLEIN_SNAPSHOT")
    if env:
        return env
    cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    d = os.path.join(cache, "models--" + HF_REPO.replace("/", "--"), "snapshots", HF_REVISION)
    if os.path.isdir(d):
        return d
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=HF_REPO, revision=HF_REVISION)


def text_encoder_dir() -> str:
    """klein's text_encoder, or (env KLEIN_TEXT_ENCODER) the bit-identical Qwen/Qwen3-4B checkpoint."""
    return os.environ.get("KLEIN_TEXT_ENCODER") or os.path.join(snapshot_dir(), "text_encoder")

# SPDX-License-Identifier: Apache-2.0
"""Model-independent host helpers shared by the single-p100a ports.

Extracted verbatim from the Z-Image-Turbo port (`tt-z-image-turbo`, Apache-2.0), where these
same functions prepare checkpoint tensors for ttnn: row padding, the matmul layout, fused qkv,
the adjacent-pair head permutation ttnn's rotary kernel expects, and the SwiGLU interleave.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import torch
from safetensors import safe_open

TILE = 32
SEQ_MULTI_OF = 32

def round_up(n: int, m: int = SEQ_MULTI_OF) -> int:
    return (n + m - 1) // m * m


# ----------------------------------------------------------------------------- checkpoint access
class LazyCheckpoint:
    """safetensors shards of one sub-folder, opened lazily; get() returns an owned tensor."""

    def __init__(self, root: str, subfolder: str):
        self.root = os.path.join(root, subfolder)
        idx = [f for f in os.listdir(self.root) if f.endswith(".safetensors.index.json")]
        if idx:
            weight_map = json.load(open(os.path.join(self.root, idx[0])))["weight_map"]
            self.key_to_file = {k: os.path.join(self.root, v) for k, v in weight_map.items()}
        else:
            (single,) = [f for f in os.listdir(self.root) if f.endswith(".safetensors")]
            path = os.path.join(self.root, single)
            with safe_open(path, framework="pt") as f:
                self.key_to_file = {k: path for k in f.keys()}
        self._handles: Dict[str, object] = {}

    def _handle(self, path):
        if path not in self._handles:
            self._handles[path] = safe_open(path, framework="pt", device="cpu")
        return self._handles[path]

    def keys(self):
        return self.key_to_file.keys()

    def get(self, key: str, dtype: torch.dtype | None = torch.bfloat16) -> torch.Tensor:
        t = self._handle(self.key_to_file[key]).get_tensor(key)
        return t.to(dtype) if dtype is not None and t.dtype != dtype else t.clone()

    def get_rows(self, key: str, rows: torch.Tensor) -> torch.Tensor:
        sl = self._handle(self.key_to_file[key]).get_slice(key)
        return torch.stack([sl[i : i + 1][0] for i in rows.reshape(-1).tolist()], dim=0)


# ----------------------------------------------------------------------------- weight layout helpers
def linear_to_mm(w: torch.Tensor) -> torch.Tensor:
    """nn.Linear weight [out, in] -> matmul weight [in, out]."""
    return w.t().contiguous()


def fuse_qkv(wq, wk, wv) -> torch.Tensor:
    return torch.cat([linear_to_mm(wq), linear_to_mm(wk), linear_to_mm(wv)], dim=1).contiguous()


def swiglu_interleave(gate_out_in: torch.Tensor, up_out_in: torch.Tensor, tile: int = TILE) -> torch.Tensor:
    """[K, 2N] weight for minimal_matmul(fuse_swiglu=True): column tile 2p = gate tile p, 2p+1 = up tile p."""
    g, u = linear_to_mm(gate_out_in), linear_to_mm(up_out_in)
    K, N = g.shape
    assert N % tile == 0, N
    return torch.stack([g.view(K, N // tile, tile), u.view(K, N // tile, tile)], dim=2).reshape(K, 2 * N).contiguous()


def interleave_pairs_permutation(head_dim: int) -> torch.Tensor:
    """new[j] = old[p[j]]: llama rotate_half pairs (i, i + D/2) -> adjacent pairs (2i, 2i + 1)."""
    half = head_dim // 2
    p = torch.empty(head_dim, dtype=torch.long)
    p[0::2] = torch.arange(half)
    p[1::2] = torch.arange(half) + half
    return p


def permute_heads_rows(w_out_in: torch.Tensor, n_heads: int, head_dim: int, perm: torch.Tensor) -> torch.Tensor:
    out, inp = w_out_in.shape
    assert out == n_heads * head_dim
    return w_out_in.view(n_heads, head_dim, inp)[:, perm, :].reshape(out, inp).contiguous()


def rot_transformation_mat(tile: int = TILE) -> torch.Tensor:
    """[1, 1, 32, 32] T with (x @ T)[2k] = -x[2k+1], (x @ T)[2k+1] = x[2k] (adjacent-pair rotation)."""
    m = torch.zeros(1, 1, tile, tile)
    m[..., torch.arange(0, tile, 2), torch.arange(1, tile, 2)] = 1.0
    m[..., torch.arange(1, tile, 2), torch.arange(0, tile, 2)] = -1.0
    return m


def pad_rows(t: torch.Tensor, rows: int) -> torch.Tensor:
    if t.shape[-2] == rows:
        return t
    out = torch.zeros(*t.shape[:-2], rows, t.shape[-1], dtype=t.dtype)
    out[..., : t.shape[-2], :] = t
    return out


# ----------------------------------------------------------------------------- geometry

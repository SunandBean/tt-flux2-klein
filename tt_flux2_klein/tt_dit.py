# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""FLUX.2 [klein] 4B transformer on ONE Blackhole (P100a), TTNN, batch 1. Mirrors ref_algo.RefDiT op for op.

Built from the same Apache-2.0 Tenstorrent patterns as the tt-z-image-turbo port's tt_dit.py (Qwen-Image-2.1 single-chip
port): fused QKV minimal_matmul, per-head RMSNorm, adjacent-pair rotary_embedding_llama, fused SwiGLU.
tt-metal's tt_dit Flux2Transformer targets TP/SP meshes (ring/joint SDPA, CCL buffers) and asserts an
unpadded spatial sequence without SP, so it is not reused here.

Layout: one joint sequence [text (512) ; image (N) ; references (R) ; pad] for every attention. Pad keys
are masked by an additive bias (only when N + R is not a tile multiple); pad queries are discarded.
Double blocks keep separate text / image streams (own projections, own LayerNorm modulation) and attend
jointly; single blocks run on the concatenation. Modulation is global per step: host-computed
(1 + scale, shift) rows are passed as layer_norm weight / bias and gates as broadcast rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import ttnn

from . import host
from .config import DIT, DiTConfig

try:
    from models.tt_dit.utils.matmul import get_matmul_config
except ImportError:  # pragma: no cover
    get_matmul_config = None

MEM = ttnn.DRAM_MEMORY_CONFIG



def _as_4d(t):
    """minimal_matmul / concatenate_heads may hand back [1, S, D]; every slice/concat here assumes [1, 1, S, D]."""
    if len(t.shape) == 4:
        return t
    shape = list(t.shape)
    return ttnn.reshape(t, [1] * (4 - len(shape)) + shape)

@dataclass
class DiTPrecision:
    weight_dtype: ttnn.DataType = ttnn.bfloat16
    mm_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    sdpa_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4  # Z-Image: fp32-acc HiFi4 SDPA held parity
    sdpa_fp32_acc: bool = True


def _dev(dev, t: torch.Tensor, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=MEM)


def _row(dev, v: torch.Tensor):
    return _dev(dev, v.to(torch.bfloat16).reshape(1, 1, 1, -1))


class DoubleBlock:
    def __init__(self, dev, ckpt, i: int, prec: DiTPrecision, cfg: DiTConfig):
        g = lambda k: ckpt.get(f"transformer_blocks.{i}.{k}", torch.bfloat16)
        wd, m = prec.weight_dtype, cfg.mlp_hidden
        a = lambda n: g(f"attn.{n}.weight")
        self.wqkv_img = _dev(dev, host.fuse_qkv(a("to_q"), a("to_k"), a("to_v")), wd)
        self.wqkv_txt = _dev(dev, host.fuse_qkv(a("add_q_proj"), a("add_k_proj"), a("add_v_proj")), wd)
        self.norm_q, self.norm_k = _row(dev, a("norm_q")), _row(dev, a("norm_k"))
        self.norm_added_q, self.norm_added_k = _row(dev, a("norm_added_q")), _row(dev, a("norm_added_k"))
        self.wo_img = _dev(dev, host.linear_to_mm(g("attn.to_out.0.weight")), wd)
        self.wo_txt = _dev(dev, host.linear_to_mm(a("to_add_out")), wd)
        ff_in, ctx_in = g("ff.linear_in.weight"), g("ff_context.linear_in.weight")
        self.ff_in_img = _dev(dev, host.swiglu_interleave(ff_in[:m], ff_in[m:]), wd)  # Flux2SwiGLU: silu(x1) * x2
        self.ff_in_txt = _dev(dev, host.swiglu_interleave(ctx_in[:m], ctx_in[m:]), wd)
        self.ff_out_img = _dev(dev, host.linear_to_mm(g("ff.linear_out.weight")), wd)
        self.ff_out_txt = _dev(dev, host.linear_to_mm(g("ff_context.linear_out.weight")), wd)


class SingleBlock:
    def __init__(self, dev, ckpt, i: int, prec: DiTPrecision, cfg: DiTConfig):
        g = lambda k: ckpt.get(f"single_transformer_blocks.{i}.attn.{k}", torch.bfloat16)
        wd, d, m = prec.weight_dtype, cfg.dim, cfg.mlp_hidden
        proj = g("to_qkv_mlp_proj.weight")  # rows: q | k | v | mlp gate | mlp up
        self.wqkv = _dev(dev, host.linear_to_mm(proj[: 3 * d]), wd)
        self.w_mlp = _dev(dev, host.swiglu_interleave(proj[3 * d: 3 * d + m], proj[3 * d + m:]), wd)
        out = g("to_out.weight")  # columns: attention (dim) | mlp (mlp_hidden)
        self.wo_attn = _dev(dev, host.linear_to_mm(out[:, :d]), wd)
        self.wo_mlp = _dev(dev, host.linear_to_mm(out[:, d:]), wd)
        self.norm_q, self.norm_k = _row(dev, g("norm_q.weight")), _row(dev, g("norm_k.weight"))


@dataclass
class Prepared:
    geo: host.Geometry
    cos: ttnn.Tensor
    sin: ttnn.Tensor
    mask: Optional[ttnn.Tensor]


class KleinDiT:
    def __init__(self, dev, ckpt, prec: Optional[DiTPrecision] = None, cfg: DiTConfig = DIT):
        self.dev, self.cfg = dev, cfg
        self.prec = prec or DiTPrecision()
        self.grid = dev.compute_with_storage_grid_size()
        g = lambda k: ckpt.get(k, torch.bfloat16)
        self.double = [DoubleBlock(dev, ckpt, i, self.prec, cfg) for i in range(cfg.n_double)]
        self.single = [SingleBlock(dev, ckpt, i, self.prec, cfg) for i in range(cfg.n_single)]
        self.x_w = _dev(dev, host.linear_to_mm(g("x_embedder.weight")))
        self.ctx_w = _dev(dev, host.linear_to_mm(g("context_embedder.weight")))
        self.proj_out = _dev(dev, host.linear_to_mm(g("proj_out.weight")))
        self.trans_mat = _dev(dev, host.rot_transformation_mat())
        self.time = host.TimeConditioning(ckpt, cfg)
        arch = dev.arch()
        ck = lambda fid, fp32: ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False)
        self.ck_mm = ck(self.prec.mm_fidelity, True)
        self.ck_norm = ck(ttnn.MathFidelity.HiFi4, True)
        self.ck_sdpa = ck(self.prec.sdpa_fidelity, self.prec.sdpa_fp32_acc)
        self.ck_rope = ck(ttnn.MathFidelity.HiFi4, True)
        self._mm_cfg_cache: Dict[tuple, object] = {}
        self.sdpa_chunks = (256, 128, 64, 32)

    # ------------------------------------------------------------------ geometry / conditioning
    def prepare(self, geo: host.Geometry) -> Prepared:
        cos, sin = host.cos_sin(host.joint_ids(geo))
        mask = host.key_mask(geo)
        f = lambda t: _dev(self.dev, t.reshape(1, 1, t.shape[0], -1))
        return Prepared(geo, f(cos), f(sin), None if mask is None else _dev(self.dev, mask))

    def release(self, prep: Prepared):
        for t in (prep.cos, prep.sin, prep.mask):
            if t is not None:
                ttnn.deallocate(t)

    def condition(self, t_sched: float) -> Dict[str, Dict[str, ttnn.Tensor]]:
        c = self.time.step(t_sched)
        rows = lambda d: {k: _row(self.dev, v) for k, v in d.items()}
        return {"img": rows(c.img), "txt": rows(c.txt), "single": rows(c.single),
                "out": rows({"w": c.out_w, "b": c.out_b})}

    @staticmethod
    def free_condition(cond):
        for group in cond.values():
            for t in group.values():
                ttnn.deallocate(t)

    # ------------------------------------------------------------------ ops
    def _mm(self, x, w, M, K, N, fuse_swiglu=False):
        key = (M, K, N)
        if key not in self._mm_cfg_cache:
            self._mm_cfg_cache[key] = get_matmul_config(M, K, N, self.grid)
        out = ttnn.experimental.minimal_matmul(x, w, config=self._mm_cfg_cache[key], compute_kernel_config=self.ck_mm,
                                               dtype=ttnn.bfloat16, memory_config=MEM, fuse_swiglu=fuse_swiglu)
        return _as_4d(out)

    def _ln(self, x, w, b):
        return ttnn.layer_norm(x, epsilon=self.cfg.eps, weight=w, bias=b, compute_kernel_config=self.ck_norm,
                               memory_config=MEM)

    def _rms(self, x, w):
        return ttnn.rms_norm(x, epsilon=self.cfg.eps, weight=w, compute_kernel_config=self.ck_norm, memory_config=MEM)

    def _gated_add(self, x, y, gate):
        gy = ttnn.multiply(y, gate, memory_config=MEM)
        ttnn.deallocate(y)
        out = ttnn.add(x, gy, memory_config=MEM)
        ttnn.deallocate(gy)
        ttnn.deallocate(x)
        return out

    def _heads(self, qkv, S):
        c = self.cfg
        if len(qkv.shape) != 4:
            qkv = ttnn.reshape(qkv, [1, 1, S, 3 * c.dim])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=c.heads, num_kv_heads=c.heads,
                                                         transpose_k_heads=False, memory_config=MEM)
        ttnn.deallocate(qkv)
        return q, k, v

    def _sdpa(self, q, k, v, prep: Prepared):
        """RoPE on the joint q/k, then attention over [text ; spatial] with pad keys masked."""
        S = prep.geo.n_joint
        qr = ttnn.experimental.rotary_embedding_llama(q, prep.cos, prep.sin, self.trans_mat, is_decode_mode=False,
                                                      compute_kernel_config=self.ck_rope)
        kr = ttnn.experimental.rotary_embedding_llama(k, prep.cos, prep.sin, self.trans_mat, is_decode_mode=False,
                                                      compute_kernel_config=self.ck_rope)
        for t in (q, k):
            ttnn.deallocate(t)
        chunk = next(ch for ch in self.sdpa_chunks if S % ch == 0)
        cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=self.grid, q_chunk_size=chunk, k_chunk_size=chunk,
                                     exp_approx_mode=False)
        attn = ttnn.transformer.scaled_dot_product_attention(
            qr, kr, v, attn_mask=prep.mask, is_causal=False, scale=self.cfg.head_dim ** -0.5, program_config=cfg,
            compute_kernel_config=self.ck_sdpa, memory_config=MEM)
        for t in (qr, kr, v):
            ttnn.deallocate(t)
        out = _as_4d(ttnn.transformer.concatenate_heads(attn, memory_config=MEM))  # [1, 1, S, dim]
        ttnn.deallocate(attn)
        return out

    def _rows(self, x, start, stop):
        x = _as_4d(x)
        return ttnn.slice(x, [0, 0, start, 0], [1, 1, stop, x.shape[-1]], memory_config=MEM)

    def _double(self, x, txt, blk: DoubleBlock, cond, prep: Prepared):
        c, geo = self.cfg, prep.geo
        Np, T = geo.n_spatial_pad, geo.n_txt
        im, tx = cond["img"], cond["txt"]
        hi = self._ln(x, im["ln1_w"], im["ln1_b"])
        ht = self._ln(txt, tx["ln1_w"], tx["ln1_b"])
        qi, ki, vi = self._heads(self._mm(hi, blk.wqkv_img, Np, c.dim, 3 * c.dim), Np)
        qt, kt, vt = self._heads(self._mm(ht, blk.wqkv_txt, T, c.dim, 3 * c.dim), T)
        for t in (hi, ht):
            ttnn.deallocate(t)
        normed = []
        for t, w in ((qt, blk.norm_added_q), (qi, blk.norm_q), (kt, blk.norm_added_k), (ki, blk.norm_k)):
            normed.append(self._rms(t, w))
            ttnn.deallocate(t)
        q = ttnn.concat(normed[0:2], dim=2, memory_config=MEM)
        k = ttnn.concat(normed[2:4], dim=2, memory_config=MEM)
        v = ttnn.concat([vt, vi], dim=2, memory_config=MEM)
        for t in (*normed, vt, vi):
            ttnn.deallocate(t)
        o = self._sdpa(q, k, v, prep)
        o_txt, o_img = self._rows(o, 0, T), self._rows(o, T, T + Np)
        ttnn.deallocate(o)
        x = self._gated_add(x, self._mm(o_img, blk.wo_img, Np, c.dim, c.dim), im["gate1"])
        txt = self._gated_add(txt, self._mm(o_txt, blk.wo_txt, T, c.dim, c.dim), tx["gate1"])
        for t in (o_txt, o_img):
            ttnn.deallocate(t)
        for stream, rows, n, w_in, w_out, M in ((x, im, "img", blk.ff_in_img, blk.ff_out_img, Np),
                                               (txt, tx, "txt", blk.ff_in_txt, blk.ff_out_txt, T)):
            h = self._ln(stream, rows["ln2_w"], rows["ln2_b"])
            m = self._mm(h, w_in, M, c.dim, 2 * c.mlp_hidden, fuse_swiglu=True)
            ttnn.deallocate(h)
            y = self._mm(m, w_out, M, c.mlp_hidden, c.dim)
            ttnn.deallocate(m)
            if n == "img":
                x = self._gated_add(stream, y, rows["gate2"])
            else:
                txt = self._gated_add(stream, y, rows["gate2"])
        return x, txt

    def _single(self, u, blk: SingleBlock, cond, prep: Prepared):
        c, S = self.cfg, prep.geo.n_joint
        s = cond["single"]
        h = self._ln(u, s["ln_w"], s["ln_b"])
        q, k, v = self._heads(self._mm(h, blk.wqkv, S, c.dim, 3 * c.dim), S)
        mlp = self._mm(h, blk.w_mlp, S, c.dim, 2 * c.mlp_hidden, fuse_swiglu=True)
        ttnn.deallocate(h)
        qn, kn = self._rms(q, blk.norm_q), self._rms(k, blk.norm_k)
        for t in (q, k):
            ttnn.deallocate(t)
        o = self._sdpa(qn, kn, v, prep)
        y_attn = self._mm(o, blk.wo_attn, S, c.dim, c.dim)
        ttnn.deallocate(o)
        y_mlp = self._mm(mlp, blk.wo_mlp, S, c.mlp_hidden, c.dim)
        ttnn.deallocate(mlp)
        y = ttnn.add(y_attn, y_mlp, memory_config=MEM)
        for t in (y_attn, y_mlp):
            ttnn.deallocate(t)
        return self._gated_add(u, y, s["gate"])

    # ------------------------------------------------------------------ public
    def context(self, prompt_embeds: torch.Tensor) -> ttnn.Tensor:
        """[512, 7680] host prompt embeddings -> context_embedder output [1, 1, 512, dim] on device."""
        p = _dev(self.dev, prompt_embeds.to(torch.bfloat16).reshape(1, 1, prompt_embeds.shape[0], -1))
        out = self._mm(p, self.ctx_w, p.shape[-2], self.cfg.joint_attention_dim, self.cfg.dim)
        ttnn.deallocate(p)
        return out

    def step(self, packed: torch.Tensor, refs: Optional[torch.Tensor], ctx: ttnn.Tensor, cond, prep: Prepared,
             taps: Optional[list] = None) -> torch.Tensor:
        """packed [N, 128] (+ refs [R, 128]) host tokens -> velocity [N, 128] float32 on host."""
        c, geo = self.cfg, prep.geo
        spatial = packed if refs is None else torch.cat([packed, refs])
        p = _dev(self.dev, host.pad_rows(spatial.to(torch.bfloat16), geo.n_spatial_pad).reshape(1, 1, geo.n_spatial_pad, -1))
        x = self._mm(p, self.x_w, geo.n_spatial_pad, c.in_channels, c.dim)
        ttnn.deallocate(p)
        txt = ttnn.clone(ctx, memory_config=MEM)
        tap = (lambda t, a=0, b=None: taps.append(ttnn.to_torch(t)[0, 0, a:b].float())) if taps is not None else (lambda *a, **k: None)
        for blk in self.double:
            x, txt = self._double(x, txt, blk, cond, prep)
            tap(x)
        u = ttnn.concat([txt, x], dim=2, memory_config=MEM)
        for t in (txt, x):
            ttnn.deallocate(t)
        for blk in self.single:
            u = self._single(u, blk, cond, prep)
            tap(u, geo.n_txt)
        xi = self._rows(u, geo.n_txt, geo.n_joint)
        ttnn.deallocate(u)
        xn = self._ln(xi, cond["out"]["w"], cond["out"]["b"])
        ttnn.deallocate(xi)
        out = self._mm(xn, self.proj_out, geo.n_spatial_pad, c.dim, c.in_channels)
        ttnn.deallocate(xn)
        host_out = ttnn.to_torch(out)[0, 0, : geo.n_img].float()
        ttnn.deallocate(out)
        return host_out

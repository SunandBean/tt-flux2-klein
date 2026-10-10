# SPDX-License-Identifier: Apache-2.0
"""Torch implementation of exactly the algorithm the TT port runs (joint [text; image; refs; pad] sequence
with masked pad keys, adjacent-pair RoPE tables, LayerNorm modulation folded into weight/bias, prompt
context_embedder once per prompt). CPU proof that the host transformations reproduce diffusers.

RefTextEncoder mirrors the TT Qwen3 prefill (27 layers, causal + padding mask, permuted q/k rows)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import host
from .config import DIT, TE, DiTConfig, TextEncoderConfig


def rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(torch.bfloat16)


def ln_mod(x, w, b, eps):
    return (F.layer_norm(x.float(), (x.shape[-1],), eps=eps) * w.float() + b.float()).to(torch.bfloat16)


class RefDiT:
    def __init__(self, ckpt, cfg: DiTConfig = DIT):
        self.cfg = cfg
        self.g = lambda k: ckpt.get(k, torch.bfloat16)
        self.time = host.TimeConditioning(ckpt, cfg)

    def _heads(self, x):
        return x.view(x.shape[0], self.cfg.heads, self.cfg.head_dim).transpose(0, 1)  # [H, S, D]

    def _attend(self, q, k, v, geo, cos, sin, mask):
        q = host.apply_rope_adjacent(q.float(), cos.float(), sin.float()).to(torch.bfloat16)
        k = host.apply_rope_adjacent(k.float(), cos.float(), sin.float()).to(torch.bfloat16)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=None if mask is None else mask[0, 0])
        return o.transpose(0, 1).reshape(q.shape[1], -1)

    def context(self, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """[512, 7680] -> [512, dim] (once per prompt)."""
        return F.linear(prompt_embeds.to(torch.bfloat16), self.g("context_embedder.weight"))

    def step(self, packed: torch.Tensor, refs: torch.Tensor | None, txt: torch.Tensor, cond: host.StepCond,
             geo: host.Geometry, taps=None) -> torch.Tensor:
        """packed [N, 128] (+ refs [R, 128]) -> velocity [N, 128] for one timestep."""
        c, g, eps = self.cfg, self.g, self.cfg.eps
        spatial = packed if refs is None else torch.cat([packed, refs])
        x = F.linear(host.pad_rows(spatial.to(torch.bfloat16), geo.n_spatial_pad), g("x_embedder.weight"))
        cos, sin = host.cos_sin(host.joint_ids(geo))
        cos, sin = cos[None], sin[None]
        mask = host.key_mask(geo)
        T = geo.n_txt
        tap = (lambda t: taps.append(t.float().clone())) if taps is not None else (lambda t: None)
        for i in range(c.n_double):
            p = f"transformer_blocks.{i}"
            a = lambda n: g(f"{p}.attn.{n}.weight")
            hx = ln_mod(x, cond.img["ln1_w"], cond.img["ln1_b"], eps)
            ht = ln_mod(txt, cond.txt["ln1_w"], cond.txt["ln1_b"], eps)
            qi, ki, vi = (self._heads(F.linear(hx, a(n))) for n in ("to_q", "to_k", "to_v"))
            qt, kt, vt = (self._heads(F.linear(ht, a(n))) for n in ("add_q_proj", "add_k_proj", "add_v_proj"))
            qi, ki = rms(qi, a("norm_q"), eps), rms(ki, a("norm_k"), eps)
            qt, kt = rms(qt, a("norm_added_q"), eps), rms(kt, a("norm_added_k"), eps)
            o = self._attend(torch.cat([qt, qi], 1), torch.cat([kt, ki], 1), torch.cat([vt, vi], 1), geo, cos, sin, mask)
            x = x + cond.img["gate1"] * F.linear(o[T:], g(f"{p}.attn.to_out.0.weight"))
            txt = txt + cond.txt["gate1"] * F.linear(o[:T], a("to_add_out"))
            ffi = lambda y, n: F.linear(swiglu(F.linear(y, g(f"{p}.{n}.linear_in.weight"))), g(f"{p}.{n}.linear_out.weight"))
            x = x + cond.img["gate2"] * ffi(ln_mod(x, cond.img["ln2_w"], cond.img["ln2_b"], eps), "ff")
            txt = txt + cond.txt["gate2"] * ffi(ln_mod(txt, cond.txt["ln2_w"], cond.txt["ln2_b"], eps), "ff_context")
            tap(x)
        u = torch.cat([txt, x])
        for i in range(c.n_single):
            p = f"single_transformer_blocks.{i}.attn"
            h = ln_mod(u, cond.single["ln_w"], cond.single["ln_b"], eps)
            proj = F.linear(h, g(f"{p}.to_qkv_mlp_proj.weight"))
            qkv, mlp = proj.split([3 * c.dim, 2 * c.mlp_hidden], dim=-1)
            q, k, v = (self._heads(t) for t in qkv.chunk(3, dim=-1))
            q, k = rms(q, g(f"{p}.norm_q.weight"), eps), rms(k, g(f"{p}.norm_k.weight"), eps)
            o = self._attend(q, k, v, geo, cos, sin, mask)
            u = u + cond.single["gate"] * F.linear(torch.cat([o, swiglu(mlp)], -1), g(f"{p}.to_out.weight"))
            tap(u[T:])
        x = ln_mod(u[T: T + geo.n_img], cond.out_w, cond.out_b, eps)
        return F.linear(x, g("proj_out.weight"))


def swiglu(x):
    a, b = x.chunk(2, dim=-1)
    return F.silu(a) * b


class RefTextEncoder:
    """Qwen3 prefill of the port: permuted q/k rows + adjacent RoPE, 27 layers, taps after 9/18/27 layers."""

    def __init__(self, ckpt, cfg: TextEncoderConfig = TE, prefix: str = "model."):
        self.cfg, self.prefix = cfg, prefix
        self.g = lambda k: ckpt.get(prefix + k, torch.bfloat16)
        self.perm = host.interleave_pairs_permutation(cfg.head_dim)

    def encode(self, ids: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        c, g, perm = self.cfg, self.g, self.perm
        L = ids.shape[0]
        h = g("embed_tokens.weight")[ids]
        cos, sin = host.te_cos_sin(L)
        mask = host.te_mask(valid)[0, 0]
        taps = []
        for li in range(c.num_layers):
            p = f"layers.{li}."
            x = rms(h, g(p + "input_layernorm.weight"), c.rms_eps)
            q = F.linear(x, host.permute_heads_rows(g(p + "self_attn.q_proj.weight"), c.heads, c.head_dim, perm))
            k = F.linear(x, host.permute_heads_rows(g(p + "self_attn.k_proj.weight"), c.kv_heads, c.head_dim, perm))
            v = F.linear(x, g(p + "self_attn.v_proj.weight"))
            q = rms(q.view(L, c.heads, c.head_dim), g(p + "self_attn.q_norm.weight")[perm], c.rms_eps).transpose(0, 1)
            k = rms(k.view(L, c.kv_heads, c.head_dim), g(p + "self_attn.k_norm.weight")[perm], c.rms_eps).transpose(0, 1)
            v = v.view(L, c.kv_heads, c.head_dim).transpose(0, 1)
            q = host.apply_rope_adjacent(q.float(), cos.float(), sin.float()).to(torch.bfloat16)
            k = host.apply_rope_adjacent(k.float(), cos.float(), sin.float()).to(torch.bfloat16)
            rep = c.heads // c.kv_heads
            o = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0), attn_mask=mask)
            h = h + F.linear(o.transpose(0, 1).reshape(L, -1), g(p + "self_attn.o_proj.weight"))
            x = rms(h, g(p + "post_attention_layernorm.weight"), c.rms_eps)
            h = h + F.linear(swiglu(torch.cat([F.linear(x, g(p + "mlp.gate_proj.weight")),
                                                F.linear(x, g(p + "mlp.up_proj.weight"))], -1)),
                             g(p + "mlp.down_proj.weight"))
            if li + 1 in c.taps:
                taps.append(h)
        return torch.cat(taps, dim=-1)  # [L, 3 * hidden] == stack(dim=1).permute.reshape of the pipeline

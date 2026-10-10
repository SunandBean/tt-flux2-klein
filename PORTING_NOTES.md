# Porting notes — FLUX.2 [klein] 4B to one Blackhole p100a

Translated and condensed from the working notes kept during the port (2026-09-26).

Target: `black-forest-labs/FLUX.2-klein-4B@e7b7dc27f91deacad38e78976d1f2b499d76a294` (Apache-2.0).
Reference implementation: diffusers 0.37.1 `Flux2KleinPipeline`.

## Upstream behaviour (diffusers 0.37.1, batch 1)

- **Prompt.** Chat template (`enable_thinking=False`, `add_generation_prompt=True`), then
  `padding="max_length"`, `max_length=512`. Qwen3-4B runs with a causal + padding-key attention mask,
  and hidden states `[9]`, `[18]`, `[27]` (the residual after layers 9, 18 and 27, with no final norm)
  are concatenated per token into `[512, 7680]`. **All 512 rows, padding included**, go into the DiT,
  and the DiT attends to all of them without a mask. Layers 28–36 are never used.
- **Token ids `(t, h, w, l)`.** Text `(0,0,0,i)`, image `(0,h,w,0)`, reference *i* `(10+10i, h, w, 0)`.
  Attention order is `[text; image; reference]`.
- **RoPE.** 32 dims per axis, theta 2000, `repeat_interleave_real=True` (adjacent pairs) — the same
  layout as ttnn's `rotary_embedding_llama`.
- **Latents.** `(1, 128, H/16, W/16)` bf16 noise generated directly (32 channels × a 2×2 patch), laid
  out row-major as `[N, 128]`. At the end they are denormalised with the VAE BatchNorm running mean and
  variance, unpatchified, then `post_quant_conv` and the decoder.
- **Schedule.** Sigmas `linspace(1, 1/4, 4)`, `mu = compute_empirical_mu(N, 4)` (determined by the image
  token count, so it changes with resolution), exponential time shift. All 4 steps call the DiT; the
  trailing sigma 0 is added by the scheduler. The timestep is rounded to bf16, divided by 1000, then
  multiplied by 1000 again inside the model.
- **Guidance.** Distilled model: `guidance_scale=1.0`, no CFG and no guidance embedding.
- **Editing / multiple references.** Each reference is reduced to at most 1024² area, center-cropped to
  a multiple of 16, VAE-encoded (distribution mode), patchified, BatchNorm-normalised, packed, and
  appended after the image tokens. Only the N image tokens are read out. The output size can be given
  separately; without one it follows the first reference.
- **Blocks.** 5 dual-stream (text and image each projected and LayerNorm-modulated, joint attention)
  and 20 single-stream (QKV and the SwiGLU MLP input in one matmul, and one matmul for the output).
  Modulation (shift / scale / gate) is **global — one set per step**, not per block.

## Port design

tt_dit's `Flux2Transformer` targets a TP/SP mesh (ring and joint SDPA, CCL buffers) and will not accept
spatial sequence padding without SP (`assert sequence_1_length == spatial.shape[1]`). That is 20 files
and 9.6k lines of dependencies, so this port is a new single-chip implementation built from the same
operation set as the Z-Image port.

- All attention runs as one sequence `[text 512; image + reference padded to a multiple of 32]`, with
  only the padded keys hidden by an additive mask (-1e9). When N+R is already a multiple of 32 there is
  no mask at all.
- LayerNorm modulation folds into `layer_norm(weight=1+scale, bias=shift)`. The per-step rows are
  computed on the host in bf16, in the same order as upstream.
- Text encoder: Qwen3-4B, 27 layers, fixed 512 tokens, causal + padding additive mask, q/k rows
  permuted for adjacent-pair RoPE. The embedding lookup stays on the host.
- VAE decoder: the tt_dit `VAEDecoder` vendored for Z-Image, used unchanged. `post_quant_conv` and the
  reference VAE encode run on the host.
- Device memory (bf16, estimated): DiT about 7.4 GB, text encoder 27 layers about 5.45 GB, VAE about
  0.1 GB — roughly 13 GB of 29.4 GB.

## Verification (CPU, 2026-09-26)

- `tests/test_host.py`, 9 tests: ids, packing, patchify, RoPE and the schedule all match diffusers.
  On a small random model, `ref_algo` against a diffusers forward is PCC 0.999991 (with 0, 13, 15 and
  25 padding tokens, references included). On a small Qwen3, the text-encoder taps are PCC 0.99997
  including the padding rows.
- `check_ref_algo.py` (real weights, first DiT call): 1024² 0.99984, 848×624 (13 padding) 0.99995,
  edit 512 (32×32 reference) 0.99992. Text encoder, valid rows: 0.999997.
- `check_ref_full.py` (all 4 steps on CPU): edit 512 latents 0.997, 1024² 0.970, 848×624 0.905. Using
  the golden prompt embeddings instead: 1024² 0.998, 848×624 0.993.
- `check_te_pad_rows.py`: the text-encoder **padding-row** deviation (PCC 0.992–0.995) is the same size
  **between transformers' own eager and sdpa implementations** (0.9927 / 0.9945). Masking causally only,
  without the padding-key mask, collapses it to 0.04–0.54 — which confirms the padding-key reading is
  the right one. So the 0.905 at 848×624 is inherent variance between bf16 implementations, not a bug.
- The golden text encoder was built from `Qwen/Qwen3-4B`: 169 of 169 tensors in klein's text-encoder
  shard 2 are identical. Shard 1 is covered by its own record (`te_equality.json`).

## Device verification

`klein_device_check.py`, on the card:

| Stage | edit 512² | 848×624 | 1024² |
|---|---:|---:|---:|
| Text encoder, valid rows | 0.9999993 | 0.9999994 | 0.9999994 |
| Text encoder, all 512 rows | 0.99917 | 0.99960 | 0.99958 |
| Reference tokens | 0.99993 | — | — |
| DiT first step | 0.99967 | 0.99955 | 0.99935 |
| End to end, reference text encoder | 0.9991 | 0.9688 | 0.9942 |
| End to end, on-card text encoder | 0.9792 | 0.7809 | 0.9601 |

The pattern matches the CPU findings above: individual stages are at 0.999+, and the 4-step no-CFG
schedule amplifies the conditioning deviation into the trajectory. Substituting the CPU text encoder
recovers most of it, which places the sensitivity in the text encoder.

# tt_flux2_klein — Python reference

FLUX.2 [klein] 4B on one Tenstorrent Blackhole p100a. The Qwen3-4B text encoder, the DiT and the
VAE all run on the card; only the scheduler and the image post-processing are host.

## Install

```bash
pip install -e .                # inside a tt-metal / ttnn environment; ttnn is not on PyPI
pip install -e ".[server]"      # also fastapi / uvicorn / pydantic
```

## Weights

`black-forest-labs/FLUX.2-klein-4B`, pinned to `e7b7dc27f91deacad38e78976d1f2b499d76a294`,
downloaded into the HF cache on first use. `KLEIN_SNAPSHOT` points at a directory you manage instead.

klein's `text_encoder` is **bit-identical to `Qwen/Qwen3-4B`** (398 of 398 tensors), so
`KLEIN_TEXT_ENCODER` can point at that checkpoint if you already have it.

## API

### `open_device(trace_region_size=0, l1_small_size=98304)`

Opens a 1×1 mesh with `DispatchCoreType.WORKER`. Pair with `close_device(dev)`.

### `KleinTT(dev, snapshot=None, te_dir=None, dit_prec=None, te_prec=None, device_vae=True)`

Loads the text encoder, the DiT and (unless `device_vae=False`) the VAE onto the card.

### `KleinTT.generate(prompt, width, height, seed, images=None) -> (PIL.Image, Timing)`

| Argument | Meaning |
|---|---|
| `width`, `height` | 512–1920, multiples of 8, at most 1920×1088 pixels. |
| `images` | `None` for text to image; 1–3 `PIL.Image` for editing or reference-guided generation. They are passed at their original aspect ratio, reduced to ≤1024² area and center-cropped to a multiple of 16. |
| `seed` | `torch.manual_seed(seed)`, matching diffusers. |

Steps are fixed at 4 with `guidance_scale` 1.0 — klein is distilled, so there is no CFG and each step
is a single DiT call.

`Timing.values` holds `text_encoder_s`, `reference_encode_s`, `context_s`, `dit_s`, `vae_decode_s`.

### Lower-level entry points

| Method | |
|---|---|
| `encode(prompt)` | prompt embeddings: layers 9, 18 and 27 concatenated to `[512, 7680]` |
| `references(images)` | reference tokens and grids, with the 4-entry encode cache |
| `denoise(emb, width, height, seed, timing, ref_tokens, grids)` | the 4-step loop |
| `decode(latents)` | VAE decode on the card |

Reference encodings are cached on pixel content, last 4 kept (`REF_CACHE_SIZE`). Reusing a source
image across prompts skips the host VAE encode, about 2.9 s at 1024².

## Modules

| Module | Role |
|---|---|
| `pipeline.py` | `KleinTT`, the end-to-end loop and the reference cache |
| `host.py` | host maths: prompt layout, token ids, RoPE, the 4-step sigma schedule and `compute_empirical_mu`, patch and latent handling, reference preprocessing |
| `common.py` | model-independent checkpoint and tensor helpers, shared with the tt-z-image-turbo port |
| `tt_dit.py` | 5 dual-stream + 20 single-stream blocks on the card |
| `tt_text_encoder.py` | Qwen3-4B on the card |
| `tt_vae.py` | the VAE on the card, over `vendor_tt_dit` |
| `vendor_tt_dit/` | `tt_dit` VAE from tt-metal `01d6e7b`, with non-square slicing and GroupNorm grid fixes |
| `ref_algo.py` | torch reference implementations used by the verification scripts |
| `device.py` | open / close / DRAM stats / timing |
| `server.py` | the HTTP app |

## Things that are easy to get wrong

- Modulation is **global per step**, not per block.
- All **512 padded text rows** go into the DiT with no mask — that is upstream behaviour, reproduced here.
- Text-encoder layers 28–36 are never used.
- RoPE is 32 dims per axis at theta 2000 with `repeat_interleave_real=True` (adjacent pairs).
- Reference image *i* gets the `t` coordinate `10 + 10*i`.

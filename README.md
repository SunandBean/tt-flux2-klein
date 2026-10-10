# tt-flux2-klein

**What this port adds** — the 4-step klein port on one chip, GroupNorm-grid and non-square fixes in the `tt_dit` VAE, and the host helpers shared with the Z-Image port.
**What it builds on** — the [`changh95/qwen-image-2.1-p150`](https://huggingface.co/changh95/qwen-image-2.1-p150) single-chip port and [tt-metal](https://github.com/tenstorrent/tt-metal)'s `tt_dit` VAE, both Apache-2.0; diffusers 0.37.1 `Flux2KleinPipeline` is the reference implementation.

**FLUX.2 [klein] 4B** ported to a single **Tenstorrent Blackhole p100a**. The Qwen3-4B text encoder,
the DiT and the VAE all run on the card in TTNN.

Model card and demos: **[sunandbean/flux2-klein-p100a](https://huggingface.co/sunandbean/flux2-klein-p100a)**

| `…a wooden sign that says "OPEN DAILY"` | `A bowl of ramen with a soft-boiled egg, scallions and chashu` |
|:---:|:---:|
| ![](media/t2i_sign.png) | ![](media/t2i_food.png) |

**A 1024×1024 image in about 3.2 s** — the fastest of the ports in this set. Editing and
reference-guided generation use the same call; reusing a reference image drops to about 2.2 s.

Against an RTX 5070 Ti running the reference `Flux2KleinPipeline` at bf16, same prompt, size, seed
and 4-step schedule: **0.331 s per DiT step on the card against 0.46 s on the GPU**. Numbers and
method on the model card.

## Install

```bash
pip install -e .    # inside a tt-metal / ttnn environment; ttnn is not on PyPI
```

```python
from PIL import Image
from tt_flux2_klein import KleinTT, open_device, close_device

dev = open_device()
try:
    model = KleinTT(dev)

    image, _ = model.generate("A red fox standing in fresh snow at dawn", 1024, 1024, seed=42)

    edited, _ = model.generate("Replace the cup of coffee with a tall glass of orange juice.",
                               512, 512, seed=3, images=[Image.open("media/croissant512.png")])
finally:
    close_device(dev)
```

Or over HTTP: `uvicorn tt_flux2_klein.server:app --host 0.0.0.0 --port 20000`.

## Why this is a new implementation

tt_dit's `Flux2Transformer` targets a TP/SP mesh — ring and joint SDPA, CCL buffers — and refuses
spatial sequence padding without SP. That is 20 files and 9.6k lines of dependencies for a single-chip
run, so this port reimplements the model with the same operation set as
[tt-z-image-turbo](https://github.com/SunandBean/tt-z-image-turbo), reusing only its VAE decoder and
host helpers.

## Accuracy, honestly

Every individual stage measures 0.999 or better against diffusers on CPU. End to end, klein runs only
**4 steps with no CFG**, so a text-encoder deviation of 6e-7 per valid row is not averaged away — it
steers the trajectory, and the end-to-end latent PCC falls to 0.96 at 1024² and 0.78 at 848×624.

`experiments/check_te_pad_rows.py` shows the padding-row deviation is the same size **between
transformers' own eager and sdpa implementations** (0.9927 / 0.9945), so this is variance between bf16
implementations rather than a porting bug. The images are visually equivalent, not numerically identical.
Details in [`PORTING_NOTES.md`](PORTING_NOTES.md).

## Layout

| Path | |
|---|---|
| `tt_flux2_klein/` | the port |
| `tt_flux2_klein/common.py` | model-independent host helpers, shared with the tt-z-image-turbo port |
| `tt_flux2_klein/vendor_tt_dit/` | `tt_dit` VAE from tt-metal `01d6e7b`, with non-square slicing and GroupNorm grid fixes |
| `examples/quickstart.py` | runnable example |
| `tt-model.yaml` | the container manifest the published image is built from — `tt-model package --container tt-model.yaml` |
| `PYTHON.md` | API reference |
| `PORTING_NOTES.md` | upstream behaviour, the design, the verification tables |
| `experiments/` | the verification scripts behind the published numbers — most import this port under the name it had in the private tree it was written in, so read [`experiments/README.md`](experiments/README.md) before running them |

## Licence

Apache-2.0. The weights
([black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B),
Apache-2.0) are not redistributed here.

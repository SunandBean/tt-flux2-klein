# vendor_tt_dit

Subset of `models/tt_dit` from [tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal) at
`01d6e7bdf4a5a954100d45cd679d5d6857d43cff` (Apache-2.0, SPDX headers kept): the dependency closure of
`models/vae/vae_sd35.py` (VAEDecoder), used for the Z-Image-Turbo and FLUX.2 klein VAE decoders on one P100a.
The runtime image ships only part of tt_dit, so these files are imported from here, not from `models.tt_dit`.

Changes (marked `[port]` in the code):

- `layers/conv2d.py`: width-slice fallback for sizes missing from the square-only slice tables; prepared
  weights are rebuilt when the input size or slicing changes (upstream prepared once per process).
- `layers/normalization.py`: `GroupNorm` picks a core-grid height that divides the tile rows
  (upstream pinned 8x8, which fails on non-square sizes such as 848x624).
- `utils/tracing.py`: `from models.tt_dit.utils import tensor` -> relative import.
- empty `__init__.py` files (upstream `models/__init__.py` only toggled kernel-compile logging).

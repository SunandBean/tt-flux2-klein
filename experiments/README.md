# Verification scripts

| Script | What it checks |
|---|---|
| `make_golden.py` | Records the diffusers CPU reference the device results are compared against |
| `check_ref_algo.py` | The first DiT call against diffusers, with real weights, at three geometries |
| `check_ref_full.py` | All 4 steps on CPU, with and without the golden prompt embeddings |
| `check_ref_encode.py` | Reference-image preprocessing and VAE encode against diffusers |
| `check_te_equality.py` | That klein's `text_encoder` is bit-identical to `Qwen/Qwen3-4B` (398/398 tensors) |
| `check_te_pad_rows.py` | The padding-row deviation — and that transformers' own eager and sdpa paths differ by the same amount |
| `klein_device_check.py` | Each stage on the card and end to end, with the on-card and the reference text encoder |
| `diag_vae_encoder.py`, `tt_vae_encoder_trial.py` | Whether the reference VAE encode belongs on the card (it stayed on the host) |
| `build_stage_cases.py`, `stage_klein.py` | The 24-case quality comparison and the timings through the HTTP service |



## Running these outside the tree they were written in

These are the scripts as they were run, inside the private working tree this port was developed in.
They are published as the record behind the numbers on the model card, and most of them need two
edits before they will run from a clone of this repo:

1. **The package name.** Six of them (`check_ref_algo.py`, `check_ref_encode.py`, `check_ref_full.py`,
   `check_te_pad_rows.py`, `klein_device_check.py`, `diag_vae_encoder.py`) do
   `from flux2klein import ...`. `flux2klein` is this port, published here as **`tt_flux2_klein`**.
2. **The `sys.path` line.** The same six insert `ROOT.parents[1] / "deploy"`, the private tree's
   package directory. From a clone that is `ROOT.parent`.

`diag_vae_encoder.py` and `tt_vae_encoder_trial.py` also import `zimage`, the sibling Z-Image port,
which is its own repository: [tt-z-image-turbo](https://github.com/SunandBean/tt-z-image-turbo), where
it is `tt_z_image_turbo`. `build_stage_cases.py` imports the private deployment's web application and
`stage_klein.py` drives its HTTP service; both are records rather than things to run.
`make_golden.py` and `check_te_equality.py` run as published. The device scripts need a p100a.

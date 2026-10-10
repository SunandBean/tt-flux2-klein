"""TT FLUX.2 VAE encoder vs the host (CPU bf16) encoder: parity and first/repeat timing per reference shape.
Needs the card (run via experiments/z-image-turbo/run_device_check.py with SCRIPT=../flux2-klein/diag_vae_encoder.py).
Writes device-check/vae_encoder.json."""
import json
from pathlib import Path
import sys
import time
import traceback

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "deploy"))
sys.path.insert(0, str(ROOT))
from flux2klein import host  # noqa: E402
from flux2klein.config import snapshot_dir  # noqa: E402
from zimage.pipeline import close_device, dram_stats, open_device  # noqa: E402

OUT = ROOT / "device-check"
OUT.mkdir(exist_ok=True)
FIX = ROOT.parents[0] / "reference-image/fixtures"
# order: new shape, repeated shape, new shapes again (arbitrary upload sizes)
IMAGES = [("croissant512", ROOT / "fixtures/croissant512.png"), ("lighthouse512", FIX / "lighthouse.png"),
          ("teapot1024", FIX / "teapot.png"), ("cat1024", FIX / "cat.png"), ("screenshot", FIX / "screenshot.png"),
          ("croissant512-again", ROOT / "fixtures/croissant512.png")]


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    import ttnn
    from diffusers import AutoencoderKLFlux2
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

    from tt_vae_encoder_trial import VAEEncoder
    from zimage.vendor_tt_dit.parallel.config import VAEParallelConfig
    from zimage.vendor_tt_dit.parallel.manager import CCLManager
    from zimage.vendor_tt_dit.utils import tensor as tt_tensor

    torch.set_num_threads(16)
    vae = AutoencoderKLFlux2.from_pretrained(snapshot_dir(), subfolder="vae", torch_dtype=torch.bfloat16).eval()
    stats = host.LatentStats.from_vae(vae)
    proc = Flux2ImageProcessor(vae_scale_factor=16)
    report = {"cases": []}
    dev = open_device()
    try:
        t0 = time.perf_counter()
        ccl = CCLManager(dev, num_links=1, topology=ttnn.Topology.Linear)
        enc = VAEEncoder.from_torch(vae.encoder.float(), mesh_device=dev, parallel_config=VAEParallelConfig.from_tuple((1, 1)),
                                    ccl_manager=ccl)
        vae.encoder.to(torch.bfloat16)
        report["load_s"] = time.perf_counter() - t0
        report["dram_after_load"] = dram_stats(dev)
        for name, path in IMAGES:
            rec = {"name": name}
            try:
                tensor, grid = host.preprocess_reference(Image.open(path).convert("RGB"), proc)
                rec["input"] = list(tensor.shape[-2:])
                with torch.no_grad():
                    t0 = time.perf_counter()
                    ref = host.pack(stats.normalize(vae.encode(tensor.to(torch.bfloat16)).latent_dist.mode().float()))
                    rec["cpu_bf16_s"] = time.perf_counter() - t0
                    for run in ("first", "second"):
                        t0 = time.perf_counter()
                        tt_in = tt_tensor.from_torch(tensor.float().permute(0, 2, 3, 1), device=dev)
                        tt_out = enc.forward(tt_in)
                        moments = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2).float()
                        ttnn.deallocate(tt_out)
                        ttnn.deallocate(tt_in)
                        moments = vae.quant_conv.float()(moments) if vae.quant_conv is not None else moments
                        mode = moments[:, : moments.shape[1] // 2]
                        tok = host.pack(stats.normalize(mode))
                        rec[f"device_{run}_s"] = time.perf_counter() - t0
                    vae.quant_conv.to(torch.bfloat16) if vae.quant_conv is not None else None
                rec["pcc_vs_cpu_bf16"] = pcc(tok, ref)
                rec["max_abs"] = float((tok - ref).abs().max())
                rec["dram"] = dram_stats(dev)["allocated_bytes"]
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"[:1500]
                rec["traceback"] = traceback.format_exc()[-2500:]
            report["cases"].append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k != "traceback"}), flush=True)
            (OUT / "vae_encoder.json").write_text(json.dumps(report, indent=2))
    finally:
        (OUT / "vae_encoder.json").write_text(json.dumps(report, indent=2))
        close_device(dev)


if __name__ == "__main__":
    main()

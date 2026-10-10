"""CPU: reference VAE encode speed and parity, fp32 (first port) vs bf16 (what the diffusers pipeline runs),
against the golden edit512 reference tokens. No accelerator."""
import json
import os
from pathlib import Path
import sys
import time

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "deploy"))
from flux2klein import host  # noqa: E402
from flux2klein.config import snapshot_dir  # noqa: E402


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    from diffusers import AutoencoderKLFlux2
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

    torch.set_num_threads(int(os.environ.get("THREADS", "8")))
    gold = torch.load(ROOT / "golden/edit512/tensors.pt", weights_only=False)
    gold_ref = gold["dit0"]["hidden_states"][0, 1024:]  # [image tokens (32x32) ; reference tokens (32x32)]
    proc = Flux2ImageProcessor(vae_scale_factor=16)
    image = Image.open(ROOT / "fixtures/croissant512.png").convert("RGB")
    big = Image.open(ROOT.parents[0] / "reference-image/fixtures/teapot.png").convert("RGB")
    out = {}
    for dtype in (torch.float32, torch.bfloat16):
        vae = AutoencoderKLFlux2.from_pretrained(snapshot_dir(), subfolder="vae", torch_dtype=dtype).eval()
        stats = host.LatentStats.from_vae(vae)
        rec = {}
        for name, img in (("croissant512", image), ("teapot1024", big)):
            tensor, grid = host.preprocess_reference(img, proc)
            with torch.no_grad():
                vae.encode(tensor.to(dtype))  # warm
                t0 = time.perf_counter()
                tokens = host.pack(stats.normalize(vae.encode(tensor.to(dtype)).latent_dist.mode().float())).to(torch.bfloat16)
                rec[name] = {"s": round(time.perf_counter() - t0, 2), "grid": grid}
            if name == "croissant512" and gold_ref is not None:
                rec[name]["pcc_vs_golden"] = pcc(tokens.float(), gold_ref.reshape(tokens.shape).float())
        out[str(dtype)] = rec
        print(dtype, rec, flush=True)
    (ROOT / "golden/ref_encode_check.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

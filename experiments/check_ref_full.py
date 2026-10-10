"""CPU: the whole port algorithm (host schedule, prompt TE, reference preprocessing + VAE encode, 4 DiT steps via
ref_algo) vs the diffusers golden final latents. Sets the realistic parity bar for the device run."""
import copy  # noqa: F401
import json
import os
from pathlib import Path
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "deploy"))
from flux2klein import host, ref_algo  # noqa: E402
from flux2klein.config import snapshot_dir, text_encoder_dir  # noqa: E402

GOLDEN = ROOT / "golden"


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    from diffusers import AutoencoderKLFlux2
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from transformers import AutoTokenizer

    torch.set_num_threads(int(os.environ.get("THREADS", "16")))
    snap = snapshot_dir()
    algo = ref_algo.RefDiT(host.LazyCheckpoint(snap, "transformer"))
    te_dir = text_encoder_dir()
    te = ref_algo.RefTextEncoder(host.LazyCheckpoint(os.path.dirname(te_dir), os.path.basename(te_dir)))
    tok = AutoTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))
    vae = AutoencoderKLFlux2.from_pretrained(snap, subfolder="vae", torch_dtype=torch.bfloat16).eval()
    stats = host.LatentStats.from_vae(vae)
    proc = Flux2ImageProcessor(vae_scale_factor=16)
    out = {}
    for name in [c for c in os.environ.get("CASES", "").split(",") if c] or ["t2i848", "edit512", "t2i1024"]:
        gold = torch.load(GOLDEN / name / "tensors.pt")
        rec = json.loads((GOLDEN / name / "record.json").read_text())
        case = rec["case"]
        width, height = rec["size"]
        with torch.no_grad():
            emb = te.encode(*host.tokenize(tok, case["prompt"]))
            if os.environ.get("GOLDEN_EMB") == "1":  # isolate the DiT/schedule from text-encoder differences
                out.setdefault(name, {})["used_golden_prompt_embeds"] = True
                emb_ours, emb = emb, gold["prompt_embeds"][0]
            refs, grids = None, ()
            if "image" in case:
                tensor, grid = host.preprocess_reference(Image.open(ROOT / case["image"]).convert("RGB"), proc)
                refs = host.pack(stats.normalize(vae.encode(tensor.to(torch.bfloat16)).latent_dist.mode())).to(torch.bfloat16)
                grids = (grid,)
                out.setdefault(name, {})["ref_tokens_pcc"] = pcc(refs, gold["dit0"]["hidden_states"][0, -refs.shape[0]:])
            geo = host.Geometry(width=width, height=height, refs=grids)
            sched = host.make_scheduler(snap, geo)
            lat = host.init_noise(geo, case["seed"])[None]
            out.setdefault(name, {})["noise_matches_golden"] = bool(torch.equal(lat[0], gold["dit0"]["hidden_states"][0, : geo.n_img]))
            ctx = algo.context(emb)
            for t in sched.timesteps:
                v = algo.step(lat[0], refs, ctx, algo.time.step(float(t)), geo)
                lat = sched.step(v.to(torch.bfloat16)[None], t, lat, return_dict=False)[0]
            final = stats.denormalize(lat[0], geo)
        ids, valid = host.tokenize(tok, case["prompt"])
        mine = emb_ours if os.environ.get("GOLDEN_EMB") == "1" else emb
        out[name].update(prompt_pcc=pcc(mine, gold["prompt_embeds"][0]),
                         prompt_pad_rows_pcc=pcc(mine[~valid], gold["prompt_embeds"][0][~valid]),
                         latents_pcc=pcc(final, gold["final_latents"]))
        print(name, out[name], flush=True)
    (GOLDEN / "ref_full_check.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

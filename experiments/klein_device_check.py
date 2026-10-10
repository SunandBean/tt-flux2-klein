"""P100a parity + timing check of the single-chip FLUX.2 [klein] 4B port against the CPU goldens.

Needs exclusive use of the card: run through experiments/z-image-turbo/run_device_check.py with
SCRIPT pointing here (see the device-phase commands in the report). Writes device-check/report.json:
  * text encoder: prompt embeddings vs golden (all rows / valid rows)
  * DiT: first-step output vs golden on identical inputs (per-block taps vs ref_algo optional: TAPS=1)
  * full 4-step latents from golden embeddings and from the TT text encoder; decoded image (TT VAE)
  * timings and DRAM after load / after each case. Stages are saved as they finish."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "deploy"))
from flux2klein import host  # noqa: E402
from flux2klein.pipeline import KleinTT, Timing, close_device, dram_stats, open_device  # noqa: E402

GOLDEN = ROOT / "golden"
OUT = ROOT / "device-check"
OUT.mkdir(exist_ok=True)
report = {"status": "starting", "cases": {}}


def save():
    (OUT / "report.json").write_text(json.dumps(report, indent=2))


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    import ttnn

    cases = [c for c in os.environ.get("CASES", "").split(",") if c] or ["edit512", "t2i848", "t2i1024"]
    dev = open_device()
    try:
        t0 = time.perf_counter()
        pipe = KleinTT(dev)
        report.update(status="loaded", load_s=time.perf_counter() - t0, dram_after_load=dram_stats(dev))
        save()
        for name in cases:
            gold = torch.load(GOLDEN / name / "tensors.pt")
            rec = json.loads((GOLDEN / name / "record.json").read_text())
            case, (width, height) = rec["case"], rec["size"]
            r = report["cases"][name] = {"case": case}
            ids, valid = host.tokenize(pipe.tokenizer, case["prompt"])
            t1 = time.perf_counter()
            emb = pipe.te.encode(ids, valid)
            r["te_s"] = time.perf_counter() - t1
            r["te_pcc"] = pcc(emb, gold["prompt_embeds"][0])
            r["te_valid_rows_pcc"] = pcc(emb[valid], gold["prompt_embeds"][0][valid])
            save()

            refs, grids = None, ()
            if "image" in case:
                refs, grids = pipe.references([Image.open(ROOT / case["image"])])
                r["ref_tokens_pcc"] = pcc(refs, gold["dit0"]["hidden_states"][0, -refs.shape[0]:])
            geo = host.Geometry(width=width, height=height, refs=grids)
            d0 = gold["dit0"]
            prep = pipe.prepared(geo)
            ctx = pipe.dit.context(d0["encoder_hidden_states"][0])
            t_sched = float(d0["timestep"].reshape(-1)[0].float() * 1000)
            cond = pipe.dit.condition(t_sched)
            hs = d0["hidden_states"][0]
            t1 = time.perf_counter()
            out0 = pipe.dit.step(hs[: geo.n_img], hs[geo.n_img:] if grids else None, ctx, cond, prep)
            r["first_step_s"] = time.perf_counter() - t1
            pipe.dit.free_condition(cond)
            ttnn.deallocate(ctx)
            r["dit_step0_pcc"] = pcc(out0, d0["out"][0, : geo.n_img].float())
            save()

            for label, feats in (("golden_te", gold["prompt_embeds"][0]), ("tt_te", emb)):
                tm = Timing()
                lat, _ = pipe.denoise(feats, width, height, case["seed"], tm, refs, grids)
                t1 = time.perf_counter()
                image, decoded = pipe.decode(lat)
                tm.mark("vae_decode_s", t1)
                image.save(OUT / f"{name}-{label}.png")
                r[label] = {"latents_pcc": pcc(lat, gold["final_latents"].float()),
                            "decoded_pcc": pcc(decoded, gold["decoded"]), "timing_s": tm.values}
                save()
            r["dram_after_case"] = dram_stats(dev)
            save()
        report["status"] = "ok"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        save()
        close_device(dev)


if __name__ == "__main__":
    main()

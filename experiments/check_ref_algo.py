"""CPU: the port's algorithm (ref_algo) on the real weights vs the golden first transformer call and prompt
embeddings. Proves conditioning folding, joint layout / masking, RoPE tables and the 27-layer TE on real weights."""
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "deploy"))
from flux2klein import host, ref_algo  # noqa: E402
from flux2klein.config import snapshot_dir, text_encoder_dir  # noqa: E402

GOLDEN = Path(os.environ.get("GOLDEN_DIR", ROOT / "golden"))


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    from transformers import AutoTokenizer

    torch.set_num_threads(int(os.environ.get("THREADS", "16")))
    snap = snapshot_dir()
    algo = ref_algo.RefDiT(host.LazyCheckpoint(snap, "transformer"))
    te_dir = text_encoder_dir()
    te = ref_algo.RefTextEncoder(host.LazyCheckpoint(os.path.dirname(te_dir), os.path.basename(te_dir)))
    tok = AutoTokenizer.from_pretrained(os.path.join(snap, "tokenizer"))
    results = {}
    for case_dir in sorted(p for p in GOLDEN.iterdir() if (p / "tensors.pt").exists()):
        gold = torch.load(case_dir / "tensors.pt")
        rec = json.loads((case_dir / "record.json").read_text())
        d0 = gold["dit0"]
        r = results[case_dir.name] = {}
        t0 = time.perf_counter()
        ids, valid = host.tokenize(tok, rec["case"]["prompt"])
        with torch.no_grad():
            emb = te.encode(ids, valid)
        r["te_pcc"] = pcc(emb, gold["prompt_embeds"][0])
        r["te_valid_rows_pcc"] = pcc(emb[valid], gold["prompt_embeds"][0][valid])
        r["te_s"] = time.perf_counter() - t0
        width, height = rec["size"]
        n_img = (width // 16) * (height // 16)
        n_spatial = d0["hidden_states"].shape[1]
        refs = ()
        if n_spatial > n_img:  # one reference; its grid from the golden ids
            rid = d0["img_ids"][0, n_img:]
            refs = ((int(rid[:, 1].max()) + 1, int(rid[:, 2].max()) + 1),)
        geo = host.Geometry(width=width, height=height, refs=refs)
        hs = d0["hidden_states"][0]
        t_sched = float(d0["timestep"].reshape(-1)[0].float() * 1000)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = algo.step(hs[:n_img], hs[n_img:] if refs else None, algo.context(d0["encoder_hidden_states"][0]),
                            algo.time.step(t_sched), geo)
        ref = d0["out"][0, :n_img].float()
        r.update(dit0_pcc=pcc(out.float(), ref), max_abs=float((out.float() - ref).abs().max()), t_sched=t_sched,
                 refs=refs, pad=geo.n_pad, cpu_s=time.perf_counter() - t0,
                 ids_match=bool(torch.equal(host.joint_ids(geo)[512:512 + geo.n_spatial], d0["img_ids"][0])))
        print(json.dumps({case_dir.name: r}), flush=True)
    (GOLDEN / "ref_algo_check.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

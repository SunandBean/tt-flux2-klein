"""CPU: why do the text encoder's pad rows (kept and attended by klein's DiT) differ from transformers?
Compares transformers' hidden_states[9/18/27] with ref_algo.RefTextEncoder per tap, valid vs pad rows, for
mask variants (port: causal + padding keys; causal only) and attention precision (bf16 vs fp32 softmax path)."""
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "deploy"))
from flux2klein import host, ref_algo  # noqa: E402
from flux2klein.config import TE, snapshot_dir, text_encoder_dir  # noqa: E402


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    torch.set_num_threads(16)
    te_dir = text_encoder_dir()
    tok = AutoTokenizer.from_pretrained(os.path.join(snapshot_dir(), "tokenizer"))
    ids, valid = host.tokenize(tok, "A cozy reading nook with a cat sleeping on a knitted blanket, warm afternoon light")
    report = {"valid_tokens": int(valid.sum())}
    for attn_impl in ("sdpa", "eager"):
        model = Qwen3ForCausalLM.from_pretrained(te_dir, torch_dtype=torch.bfloat16, attn_implementation=attn_impl).eval()
        with torch.no_grad():
            hs = model(input_ids=ids[None], attention_mask=valid[None].long(), output_hidden_states=True, use_cache=False).hidden_states
        report[f"hf_{attn_impl}"] = torch.cat([hs[k][0] for k in TE.taps], -1)
        del model
    ref = ref_algo.RefTextEncoder(host.LazyCheckpoint(os.path.dirname(te_dir), os.path.basename(te_dir)))
    variants = {"port_mask": host.te_mask}
    orig = host.te_mask
    with torch.no_grad():
        report["port"] = ref.encode(ids, valid)
        host.te_mask = lambda v, dtype=torch.bfloat16: orig(torch.ones_like(v), dtype)  # causal only
        report["causal_only"] = ref.encode(ids, valid)
        host.te_mask = orig
    out = {"valid_tokens": report["valid_tokens"]}
    D = TE.hidden
    for a in ("port", "causal_only", "hf_eager"):
        for b in ("hf_sdpa",):
            if a == b:
                continue
            x, y = report[a], report[b]
            out[f"{a}_vs_{b}"] = {f"tap{k}": {"valid": pcc(x[valid, i * D:(i + 1) * D], y[valid, i * D:(i + 1) * D]),
                                               "pad": pcc(x[~valid, i * D:(i + 1) * D], y[~valid, i * D:(i + 1) * D])}
                                   for i, k in enumerate(TE.taps)}
    print(json.dumps(out, indent=1))
    (ROOT / "golden" / "te_pad_rows.json").write_text(json.dumps(out, indent=2))
    del variants


if __name__ == "__main__":
    main()

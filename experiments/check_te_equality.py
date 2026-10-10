"""CPU: are all klein text_encoder tensors bit-identical to Qwen/Qwen3-4B? Writes golden/te_equality.json."""
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

K = Path(os.environ["KLEIN_SNAPSHOT"]) / "text_encoder"
Q = Path(os.environ["QWEN3_4B"])
km = json.load(open(K / "model.safetensors.index.json"))["weight_map"]
qm = json.load(open(Q / "model.safetensors.index.json"))["weight_map"]
handles = {}
h = lambda p: handles.setdefault(p, safe_open(p, "pt"))
same, diff = 0, []
for key in sorted(km):
    a = h(str(K / km[key])).get_tensor(key)
    b = h(str(Q / qm[key])).get_tensor(key)
    if a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b):
        same += 1
    else:
        diff.append(key)
out = {"tensors": len(km), "identical": same, "different": diff, "keys_equal": set(km) == set(qm)}
print(json.dumps(out))
(Path(__file__).resolve().parent / "golden" / "te_equality.json").write_text(json.dumps(out, indent=2))

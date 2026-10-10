"""CPU golden references for the single-P100a FLUX.2 [klein] 4B port. No accelerator access.

Runs the pinned diffusers 0.37.1 Flux2KleinPipeline (bf16, 4 steps, guidance 1.0 per the model card) and
saves what a TT port must reproduce: prompt embeddings (Qwen3 layers 9/18/27, 512 tokens), the first
transformer call's inputs/outputs, final latents (after BN de-normalization + unpatchify), decoded image,
reference-image tokens for the edit case, and CPU timings.

Text encoder: until klein's text_encoder shard 1 is downloaded, TEXT_ENCODER may point at Qwen/Qwen3-4B
(shard 2 of klein's text encoder is bit-identical to it, 169/169 tensors); the record says which was used."""
import json
import os
from pathlib import Path
import time

import torch
from PIL import Image

SNAPSHOT = Path(os.environ["KLEIN_SNAPSHOT"])
TEXT_ENCODER = os.environ.get("TEXT_ENCODER", str(SNAPSHOT / "text_encoder"))
ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("GOLDEN_OUT", ROOT / "golden"))
STEPS = 4
CASES = [
    {"name": "t2i1024", "prompt": "A red fox standing in fresh snow at dawn, wildlife photography", "width": 1024,
     "height": 1024, "seed": 42},
    {"name": "t2i848", "prompt": "A cozy reading nook with a cat sleeping on a knitted blanket, warm afternoon light",
     "width": 848, "height": 624, "seed": 7},
    {"name": "edit512", "prompt": "Replace the cup of coffee with a tall glass of orange juice. Keep everything else the same.",
     "image": "fixtures/croissant512.png", "seed": 3},
]


def main():
    from diffusers import Flux2KleinPipeline
    from transformers import Qwen3ForCausalLM

    torch.set_num_threads(int(os.environ.get("THREADS", "16")))
    te = Qwen3ForCausalLM.from_pretrained(TEXT_ENCODER, torch_dtype=torch.bfloat16).eval()
    pipe = Flux2KleinPipeline.from_pretrained(SNAPSHOT, text_encoder=te, torch_dtype=torch.bfloat16)
    selected = set(filter(None, os.environ.get("CASES", "").split(",")))
    for case in CASES:
        if selected and case["name"] not in selected:
            continue
        out = OUT / case["name"]
        out.mkdir(parents=True, exist_ok=True)
        record = {"case": case, "steps": STEPS, "text_encoder": TEXT_ENCODER, "timings_s": {}}
        t0 = time.perf_counter()
        with torch.no_grad():
            prompt_embeds, _ = pipe.encode_prompt(case["prompt"], device="cpu")
        record["timings_s"]["text_encoder"] = time.perf_counter() - t0
        ids = pipe.tokenizer(pipe.tokenizer.apply_chat_template([{"role": "user", "content": case["prompt"]}],
                             tokenize=False, add_generation_prompt=True, enable_thinking=False),
                             return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        record["valid_tokens"] = int(ids["attention_mask"].sum())

        calls = []

        def hook(module, args, kwargs, output):
            if not calls:
                calls.append({k: kwargs[k].detach().clone() for k in ("hidden_states", "encoder_hidden_states",
                                                                       "timestep", "img_ids", "txt_ids")}
                             | {"out": output[0].detach().clone()})
            record["timings_s"].setdefault("dit_calls", []).append(time.perf_counter() - hook.started)
            hook.started = time.perf_counter()

        handle = pipe.transformer.register_forward_hook(hook, with_kwargs=True)
        kwargs = {}
        if "image" in case:
            kwargs["image"] = Image.open(ROOT / case["image"]).convert("RGB")
        else:
            kwargs.update(width=case["width"], height=case["height"])
        generator = torch.Generator("cpu").manual_seed(case["seed"])
        hook.started = t0 = time.perf_counter()
        latents = pipe(prompt_embeds=prompt_embeds, num_inference_steps=STEPS, guidance_scale=1.0,
                       generator=generator, output_type="latent", **kwargs).images
        record["timings_s"]["denoise"] = time.perf_counter() - t0
        handle.remove()

        t0 = time.perf_counter()
        with torch.no_grad():
            decoded = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
        record["timings_s"]["vae_decode"] = time.perf_counter() - t0
        image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        image.save(out / "image.png")
        record["size"] = list(image.size)
        record["dit_forwards"] = len(record["timings_s"].get("dit_calls", []))
        record["scheduler_timesteps"] = [float(t) for t in pipe.scheduler.timesteps]
        record["scheduler_sigmas"] = [float(s) for s in pipe.scheduler.sigmas]
        torch.save({"prompt_embeds": prompt_embeds, "dit0": calls[0], "final_latents": latents,
                    "decoded": decoded.float()}, out / "tensors.pt")
        (out / "record.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

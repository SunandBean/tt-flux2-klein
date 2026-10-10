"""Build the klein staging payloads exactly as the web + worker would (no accelerator).

Run with the project on PYTHONPATH (e.g. inside the application image). For every quality case of
experiments/reference-image/quality_cases.json the fixture goes through app.references.normalize_reference,
the prompt through effective_prompt, and the model input is what PHOTO_REFERENCE_INPUT=original sends.
Adds a text-to-image comparison set. Writes stage/cases.json (+ canvases for the copy check)."""
import base64
import io
import json
from pathlib import Path

from PIL import Image

from app.references import effective_prompt, normalize_reference

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
REF = PROJECT / "experiments/reference-image"
OUT = ROOT / "stage"
OUT.mkdir(exist_ok=True)

T2I = [
    ("t2i_portrait", "Close-up portrait of an elderly woman with silver hair laughing, window light, 85mm photograph"),
    ("t2i_product", "A sleek matte black wireless headphone on a white pedestal, studio product photography"),
    ("t2i_sign", "A cozy cafe storefront with a wooden sign that says \"OPEN DAILY\", morning light"),
    ("t2i_anime", "A girl with a red umbrella standing at a rainy train platform, anime key visual"),
    ("t2i_landscape", "Snow-capped mountains reflected in a still alpine lake at sunrise, landscape photograph"),
    ("t2i_food", "A bowl of ramen with a soft-boiled egg, scallions and chashu, overhead food photography"),
    ("t2i_interior", "A minimalist Scandinavian living room with a large window and indoor plants, architectural photo"),
    ("t2i_count", "Three red apples and two green pears on a wooden table, still life photograph"),
    ("t2i_hands", "Hands of a potter shaping a clay vase on a spinning wheel, close-up photograph"),
    ("t2i_korean", "비 오는 밤 서울 골목의 따뜻한 조명, 수채화"),
]


def main():
    cases = json.loads((REF / "quality_cases.json").read_text())["cases"]
    staged = []
    for name, prompt in T2I:
        staged.append({"id": name, "task_mode": "text_to_image", "prompt": prompt, "effective_prompt": prompt,
                       "width": 1024, "height": 1024, "seed": 1234})
    for case in cases:
        data = (REF / "fixtures" / f"{case['fixture']}.png").read_bytes()
        meta, canonical, canvas = normalize_reference(data, f"{case['fixture']}.png")
        with Image.open(io.BytesIO(canonical)) as image:  # PHOTO_REFERENCE_INPUT=original, alpha on white
            flat = Image.new("RGB", image.size, "white")
            flat.paste(image, mask=image.convert("RGBA").getchannel("A"))
            buf = io.BytesIO()
            flat.save(buf, format="PNG")
        canvas_path = OUT / f"canvas-{case['id']}.png"
        canvas_path.write_bytes(canvas)
        staged.append({"id": case["id"], "task_mode": case["task_mode"], "prompt": case["prompt"],
                       "effective_prompt": effective_prompt(case["prompt"], case["task_mode"], case.get("preserve_text", ""),
                                                            case.get("reference_elements", [])),
                       "width": case["width"], "height": case["height"], "seed": case["seed"],
                       "image_b64": base64.b64encode(buf.getvalue()).decode(), "canvas": canvas_path.name,
                       "padding": meta["padding"], "reference_size": [meta["width"], meta["height"]],
                       "visual_criteria": case.get("visual_criteria", [])})
    (OUT / "cases.json").write_text(json.dumps(staged, ensure_ascii=False))
    print(len(staged), "cases")


if __name__ == "__main__":
    main()

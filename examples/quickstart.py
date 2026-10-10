# SPDX-License-Identifier: Apache-2.0
"""Text to image and an edit on one p100a.

    python examples/quickstart.py "A red fox standing in fresh snow at dawn"
"""
import sys

from tt_flux2_klein import KleinTT, close_device, dram_stats, open_device

prompt = sys.argv[1] if len(sys.argv) > 1 else "A red fox standing in fresh snow at dawn, wildlife photography"

dev = open_device()
try:
    model = KleinTT(dev)   # downloads the snapshot into the HF cache on first use
    print(f"device DRAM {dram_stats(dev)['allocated_bytes'] / 2**30:.1f} GiB")

    image, timing = model.generate(prompt, 1024, 1024, seed=42)
    image.save("output.png")
    print({k: round(v, 3) for k, v in timing.values.items()})

    # editing: pass 1-3 reference images to the same call
    # from PIL import Image
    # edited, _ = model.generate("Replace the cup with a glass of orange juice.", 512, 512,
    #                            seed=3, images=[Image.open("../media/croissant512.png")])
finally:
    close_device(dev)

# SPDX-License-Identifier: Apache-2.0
"""FLUX.2 [klein] 4B on one Tenstorrent Blackhole p100a.

Text to image, image editing and reference-guided generation. The Qwen3-4B text encoder,
the DiT (5 dual-stream + 20 single-stream blocks) and the VAE all run on the card in TTNN.
"""
from .pipeline import KleinTT, Timing, close_device, dram_stats, open_device

__all__ = ["KleinTT", "Timing", "open_device", "close_device", "dram_stats"]
__version__ = "0.1.0"

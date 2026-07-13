"""Fold the style LoRA into the base weights and save a plain HF model.

llama.cpp can apply a LoRA at runtime, but merging first quantizes better and
keeps the serving path trivial. No GPU needed — merging is weight arithmetic
(W + BA·α/r), so this runs on CPU anywhere with enough RAM (~2 bytes/param).

Output feeds `convert_hf_to_gguf.py`. See QUANTIZATION-CHECKLIST.md.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM

BASE_ID = "google/gemma-4-E4B-it"
ADAPTER_ID = os.environ.get("HF_ADAPTER_REPO", "rocku21/gemma-4-e4b-style-captioner-lora")
OUT_DIR = Path(__file__).parent / "merged"  # never relative to the caller's cwd

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="overwrite an existing merged/ directory")
args = parser.parse_args()

# Fail fast: both repos are private or gated, and a 15 GB download is a bad
# place to discover a missing token.
if not os.environ.get("HF_TOKEN"):
    sys.exit("HF_TOKEN not set — the base model is gated and the adapter is private.")
if OUT_DIR.exists() and any(OUT_DIR.iterdir()) and not args.force:
    sys.exit(f"{OUT_DIR} already exists and is not empty — pass --force to overwrite.")

print(f"loading base {BASE_ID} (bf16, CPU)")
base = AutoModelForMultimodalLM.from_pretrained(BASE_ID, dtype=torch.bfloat16)

print(f"applying adapter {ADAPTER_ID}")
merged = PeftModel.from_pretrained(base, ADAPTER_ID).merge_and_unload()

# merge_and_unload() returns a model either way; only the absence of LoRA
# modules proves the adapter actually folded in. Silently shipping the base
# model would make every downstream measurement a lie.
leftover = [n for n, _ in merged.named_parameters() if "lora" in n.lower()]
assert not leftover, f"adapter did not merge — {len(leftover)} LoRA params survive: {leftover[:3]}"

def save_text_assets(out_dir: Path) -> None:
    """Write the tokenizer + chat template the GGUF converter needs.

    Gemma 4's AutoProcessor imports its vision stack (needs torchvision). Our
    style stage is text-only, so fall back to the tokenizer rather than require
    it — the tokenizer files are identical either way.
    """
    try:
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(BASE_ID).save_pretrained(out_dir)
    except ImportError as e:
        print(f"processor unavailable ({e.__class__.__name__}); saving tokenizer only")
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(BASE_ID).save_pretrained(out_dir)

    # chat_template.jinja is a standalone file at the repo root; whichever class
    # saved above may or may not emit it. Guarantee it lands.
    if not (out_dir / "chat_template.jinja").exists():
        from huggingface_hub import hf_hub_download

        shutil.copy(hf_hub_download(BASE_ID, "chat_template.jinja"), out_dir / "chat_template.jinja")


OUT_DIR.mkdir(parents=True, exist_ok=True)
merged.save_pretrained(OUT_DIR)
save_text_assets(OUT_DIR)

# The GGUF must carry the turn markers the model trained on. A missing template
# makes llama-server format prompts differently than training did, and the
# damage looks exactly like quantization damage — a very long wrong road.
template = OUT_DIR / "chat_template.jinja"
if not template.exists():
    sys.exit(f"no {template.name} in {OUT_DIR} — find the template before converting.")

arch = json.loads((OUT_DIR / "config.json").read_text()).get("architectures")
size_gb = sum(f.stat().st_size for f in OUT_DIR.glob("*.safetensors")) / 1e9

print(f"\nmerged -> {OUT_DIR}  ({size_gb:.1f} GB)")
print(f"architectures: {arch}   <- must be registered in llama.cpp conversion/gemma.py")
print(f"chat template: {template.name} present")
print("\nnext: python <llama.cpp>/convert_hf_to_gguf.py "
      f"{OUT_DIR} --outfile gemma4-style-f16.gguf --outtype f16")

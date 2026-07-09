import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoProcessor, AutoModelForMultimodalLM, set_seed
from trl import SFTTrainer, SFTConfig

import common  # bootstraps the project root onto sys.path
from agent.vlm import _extract_json, _is_real_caption

set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true",
                    help="tiny model + few samples; proves plumbing, trains nothing useful")
args = parser.parse_args()

if args.smoke:
    MODEL_ID = "Qwen/Qwen3-0.6B"  # tiny, text-only, ungated
    N_TRAIN, N_VAL, EPOCHS = 8, 2, 1
else:
    MODEL_ID = "google/gemma-4-E4B-it"
    N_TRAIN, N_VAL, EPOCHS = None, None, 3  # None = use everything

HF_REPO = os.environ.get("HF_ADAPTER_REPO", "")

# Fail fast: everything the real run needs, checked before any download/training.
if not args.smoke:
    if not torch.cuda.is_available():
        sys.exit("No GPU visible — this is the real run, refusing to train on CPU.")
    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN not set — the adapter could not be pushed after training.")
    if not HF_REPO:
        sys.exit("HF_ADAPTER_REPO not set, e.g. <user>/gemma-4-e4b-style-captioner-lora")
if torch.cuda.is_available():
    print(f"Training on: {torch.cuda.get_device_name(0)}")

train_path = Path(__file__).parent / "data" / "train.jsonl"
val_path = Path(__file__).parent / "data" / "val.jsonl"
train_dict = load_dataset("json", data_files={"train": str(train_path), "val": str(val_path)})

train_ds = train_dict["train"].select(range(N_TRAIN)) if N_TRAIN else train_dict["train"]
val_ds = train_dict["val"].select(range(N_VAL)) if N_VAL else train_dict["val"]

if args.smoke:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    processor = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)  # default dtype
else:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")

training_args = SFTConfig(
    output_dir=str(Path(__file__).parent / "checkpoints"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=8,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_length=1024,
    bf16=torch.cuda.is_available(),  # real run: bf16; smoke on Mac: fp32
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
    seed=42,
    assistant_only_loss=True,
)

print("--- templated sample (eyeball the turn markers) ---")
print(processor.apply_chat_template(train_ds[0]["messages"], tokenize=False))
print("----------------------------------------------------")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    peft_config=lora_config,
    processing_class=processor,
)

trainer.train()


def evaluate(m, proc, dataset, label):
    """Generate on each val row and hold the output to the production bar."""
    tok = getattr(proc, "tokenizer", proc)  # AutoProcessor wraps the tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    device = next(m.parameters()).device
    m.eval()
    rows, json_ok, styles_ok = [], 0, 0
    for i, row in enumerate(dataset):
        enc = proc.apply_chat_template(
            [row["messages"][0]], add_generation_prompt=True, return_tensors="pt"
        )
        # tensor in older transformers, BatchEncoding in newer — normalize
        ids = (enc if torch.is_tensor(enc) else enc["input_ids"]).to(device)
        with torch.no_grad():
            out = m.generate(ids, max_new_tokens=300, do_sample=True,
                             temperature=0.7, pad_token_id=pad_id)
        # generate() returns prompt + completion; keep only the new tokens
        completion = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        parsed = _extract_json(completion)
        ok_json = isinstance(parsed, dict)
        ok_styles = ok_json and all(
            isinstance(parsed.get(s), str) and _is_real_caption(parsed[s])
            for s in common.STYLES
        )
        json_ok += ok_json
        styles_ok += ok_styles
        rows.append({"idx": i, "model": label, "completion": completion,
                     "json_ok": bool(ok_json), "styles_ok": bool(ok_styles),
                     "reference": row["messages"][1]["content"]})
    n = len(dataset)
    print(f"[eval:{label}] JSON-parse {json_ok}/{n} | all-4-styles {styles_ok}/{n}")
    return rows


tuned = trainer.model
records = evaluate(tuned, processor, val_ds, "tuned")
with tuned.disable_adapter():
    records += evaluate(tuned, processor, val_ds, "base")

eval_dir = Path(__file__).parent / "eval_results"
eval_dir.mkdir(exist_ok=True)
out_path = eval_dir / ("eval_smoke.jsonl" if args.smoke else "eval_real.jsonl")
with open(out_path, "w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"eval generations -> {out_path}")

if not args.smoke:
    tuned.push_to_hub(HF_REPO, private=True)
    processor.push_to_hub(HF_REPO, private=True)
    print(f"adapter + tokenizer pushed to https://huggingface.co/{HF_REPO}")

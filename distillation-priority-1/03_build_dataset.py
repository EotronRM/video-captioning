"""Phase A step 3: optional LLM-judge filter, then emit chat-format JSONL.

Output is TRL SFTTrainer-ready:
  {"messages": [{"role": "user", "content": style_prompt(description, styles)},
                {"role": "assistant", "content": "<captions as a JSON string>"}]}

Usage:
  uv run --env-file .env python distillation-priority-1/03_build_dataset.py [--judge] [--val-frac 0.1]
"""

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import common
from agent.prompts import style_prompt
from agent.vlm import _extract_json

JUDGE_PROMPT = """Rate how well each caption matches its requested style for this video description.

Description:
{description}

Captions:
{captions}

Style definitions: formal = professional/objective; sarcastic = dry/ironic/mocking; \
humorous_tech = funny with technology or programming references; \
humorous_non_tech = funny with zero technical jargon.

Also penalize captions that contradict the description's content.
Respond with ONLY a JSON object mapping each style to a score from 0.0 to 1.0, e.g. \
{{"formal": 0.9, "sarcastic": 0.7, "humorous_tech": 0.8, "humorous_non_tech": 0.6}}."""


def judge(rec):
    try:
        raw = common.chat(
            common.DESC_MODEL,
            JUDGE_PROMPT.format(
                description=rec["description"],
                captions=json.dumps(rec["captions"], ensure_ascii=False, indent=2),
            ),
            temperature=0.0,
        )
        scores = _extract_json(raw)
        vals = [float(scores[s]) for s in common.STYLES]
        return sum(vals) / len(vals)
    except Exception as e:
        print("[%s] judge failed (%s) — keeping sample" % (rec["id"], str(e)[:80]), flush=True)
        return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true",
                    help="LLM-judge each sample; drop mean style score < --threshold")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(common.DATA_DIR / "labeled.jsonl") as f:
        records = [json.loads(line) for line in f]
    print("%d labeled records" % len(records))

    if args.judge:
        with ThreadPoolExecutor(max_workers=8) as ex:
            scores = list(ex.map(judge, records))
        before = len(records)
        records = [r for r, sc in zip(records, scores) if sc >= args.threshold]
        print("judge kept %d/%d (threshold %.2f)" % (len(records), before, args.threshold))

    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.val_frac))
    splits = {"val": records[:n_val], "train": records[n_val:]}

    for name, recs in splits.items():
        path = common.DATA_DIR / ("%s.jsonl" % name)
        with open(path, "w") as f:
            for r in recs:
                sample = {"messages": [
                    {"role": "user", "content": style_prompt(r["description"], common.STYLES)},
                    {"role": "assistant", "content": json.dumps(r["captions"], ensure_ascii=False)},
                ]}
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print("%s: %d samples -> %s" % (name, len(recs), path))

    print("category coverage:", dict(Counter(r["category"] for r in records)))


if __name__ == "__main__":
    main()

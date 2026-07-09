"""Phase A step 2: teacher-label each description with 4-style captions.

Uses the PRODUCTION style prompt (agent/prompts.py) so the student trains on
the same input distribution it will see at inference, and the agent's own
JSON validation so nothing production would reject enters the dataset.
Resumable: already-labeled ids are skipped on rerun (safe because 01 is
seeded and deterministic).

Usage:
  uv run --env-file .env python distillation-priority-1/02_teacher_label.py
"""

import json
from concurrent.futures import ThreadPoolExecutor

import common
from agent.prompts import style_prompt
from agent.vlm import _extract_json, _is_real_caption


def main():
    src = common.DATA_DIR / "descriptions.jsonl"
    out_path = common.DATA_DIR / "labeled.jsonl"
    with open(src) as f:
        descs = [json.loads(line) for line in f]

    seen = set()
    if out_path.exists():
        with open(out_path) as f:
            seen = {json.loads(line)["id"] for line in f}
        print("resuming: %d already labeled" % len(seen))
    todo = [d for d in descs if d["id"] not in seen]

    def label(rec):
        try:
            raw = common.chat(
                common.TEACHER_MODEL,
                style_prompt(rec["description"], common.STYLES),
                temperature=0.8,
            )
            parsed = _extract_json(raw)
            if not isinstance(parsed, dict):
                raise ValueError("no JSON object in teacher output")
            captions = {}
            for s in common.STYLES:
                v = parsed.get(s)
                if not (isinstance(v, str) and _is_real_caption(v)):
                    raise ValueError("style %r missing/invalid" % s)
                captions[s] = v.strip()
            return {"id": rec["id"], "category": rec["category"],
                    "description": rec["description"], "captions": captions}
        except Exception as e:
            print("[%s] rejected: %s" % (rec["id"], str(e)[:120]), flush=True)
            return None

    kept = 0
    with ThreadPoolExecutor(max_workers=8) as ex, open(out_path, "a") as f:
        for rec in ex.map(label, todo):
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
                if kept % 20 == 0:
                    print("%d/%d labeled" % (kept, len(todo)), flush=True)

    print("labeled %d new, rejected %d -> %d total in %s"
          % (kept, len(todo) - kept, len(seen) + kept, out_path))


if __name__ == "__main__":
    main()

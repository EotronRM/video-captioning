"""Generate captions for the val rows against a running llama-server.

One run per quantization level. Emits `eval_results/eval_<level>.jsonl` in the
exact row shape `05_judge_evals.py` expects, so judging needs no new code:

    llama-server -m myguff/gemma4-style-Q4_K_M.gguf -c 2048 --port 8080
    uv run python quantization-priority-3/generate_evals.py --level q4_k_m
    uv run python distillation-priority-1/05_judge_evals.py \
        --evals quantization-priority-3/eval_results/eval_q4_k_m.jsonl --vs teacher

Decode settings mirror the HF eval in `04_train_lora.py` (temp 0.7, 300 new
tokens) — otherwise we would be measuring decoding differences, not
quantization. Requests are sequential: that is what llama-server does on CPU by
default, and it is what makes the tok/s numbers mean anything.

The server applies the chat template baked into the GGUF, which is why that
metadata check was not optional.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.prompts import STYLE_SPECS  # noqa: E402  (needs the path bootstrap above)
from agent.vlm import _extract_json, _is_real_caption  # noqa: E402

STYLES = list(STYLE_SPECS)
VAL_PATH = PROJECT_ROOT / "distillation-priority-1" / "data" / "val.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "eval_results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True,
                    help="quantization level, used for the filename: bf16 | q8_0 | q4_k_m")
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--temperature", type=float, default=0.7)  # matches the HF eval
    ap.add_argument("--max-tokens", type=int, default=300)     # matches the HF eval
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="first N rows only (smoke test)")
    ap.add_argument("--thinking", action="store_true",
                    help="let Gemma 4 emit a thought channel (see the note below — it will "
                         "eat the whole token budget and return empty content)")
    args = ap.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="local")  # llama-server ignores the key

    # Fail fast on a server that is not up, rather than 19 confusing timeouts.
    try:
        served = client.models.list().data[0].id
    except Exception as e:
        sys.exit(f"no llama-server at {args.base_url} ({type(e).__name__}: {e})\n"
                 f"start it with: llama-server -m <gguf> -c 2048 --port 8080")
    print(f"server model: {served}")

    rows = [json.loads(line) for line in VAL_PATH.read_text().splitlines()]
    if args.limit:
        rows = rows[:args.limit]

    records, speeds, truncated = [], [], 0
    for idx, row in enumerate(rows):
        prompt = row["messages"][0]["content"]

        started = time.perf_counter()
        resp = client.chat.completions.create(
            model=served,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            # Gemma 4 has a thought channel, and llama-server turns it ON by default
            # (its --reasoning-budget is unlimited). Thinking burns the entire token
            # budget, leaves `content` EMPTY and finish_reason=length. Training
            # rendered prompts without <|think|>, so thinking-off is what matches
            # the weights. Same failure Kimi K2.6 gave us on day one.
            extra_body={"chat_template_kwargs": {"enable_thinking": args.thinking}},
        )
        elapsed = time.perf_counter() - started

        choice = resp.choices[0]
        completion = (choice.message.content or "").strip()
        reasoning = getattr(choice.message, "reasoning_content", None) or ""
        out_tokens = getattr(resp.usage, "completion_tokens", 0) or 0
        tok_per_s = out_tokens / elapsed if elapsed > 0 else 0.0
        speeds.append(tok_per_s)
        if choice.finish_reason == "length":
            truncated += 1

        # The production bar, not a looser one: same validators the agent ships.
        parsed = _extract_json(completion)
        ok_json = isinstance(parsed, dict)
        ok_styles = ok_json and all(
            isinstance(parsed.get(s), str) and _is_real_caption(parsed[s]) for s in STYLES
        )

        records.append({
            "idx": idx,
            "model": "tuned",  # the arm under test; 05_judge_evals.py's default --arm
            "completion": completion,
            "json_ok": bool(ok_json),
            "styles_ok": bool(ok_styles),
            "reference": row["messages"][1]["content"],
            "level": args.level,
            "latency_s": round(elapsed, 2),
            "completion_tokens": out_tokens,
            "tok_per_s": round(tok_per_s, 2),
            "finish_reason": choice.finish_reason,
            "reasoning_chars": len(reasoning),
        })
        flag = "ok " if ok_styles else ("json" if ok_json else "BAD")
        note = f"  TRUNCATED (+{len(reasoning)}ch reasoning)" if choice.finish_reason == "length" else ""
        print(f"  [{idx:2}/{len(rows)}] {flag}  {out_tokens:3} tok  "
              f"{elapsed:5.1f}s  {tok_per_s:5.1f} tok/s{note}")

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"eval_{args.level}.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(records)
    json_ok, styles_ok = sum(r["json_ok"] for r in records), sum(r["styles_ok"] for r in records)
    print(f"\n[{args.level}] JSON-parse {json_ok}/{n} | all-4-styles {styles_ok}/{n}")
    print(f"[{args.level}] median {statistics.median(speeds):.1f} tok/s | "
          f"total {sum(r['latency_s'] for r in records):.0f}s for {n} clips")
    print(f"generations -> {out_path}")

    # A wall of truncations means thinking is on, not that the model is bad.
    if truncated:
        print(f"\n!! {truncated}/{n} truncated at max_tokens={args.max_tokens}. "
              f"{'Rerun without --thinking.' if args.thinking else 'Raise --max-tokens.'}")
    if json_ok == 0:
        sys.exit("every row failed to parse — refusing to leave a file that looks like data.")


if __name__ == "__main__":
    main()

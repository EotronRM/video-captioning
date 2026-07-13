"""Judge the tuned-vs-base generations produced by 04_train_lora.py.

Runs on the Mac against the API — no GPU, no notebook clock.

Why this script exists: the training-time metric (JSON validity) saturated at
100% for BOTH arms, so it cannot discriminate. What the hackathon actually
scores is caption accuracy + style match, which needs an LLM judge.

Three design choices that make the verdict trustworthy:
  1. The judge is NOT the teacher. Kimi would reward its own distilled style;
     glm-5p2 is a different family, so self-preference is out.
  2. Candidates are re-serialized from parsed JSON before judging. Base
     pretty-prints and tuned emits one line — raw text would let the judge
     identify the arms by whitespace, and the comparison would stop being blind.
  3. Every pair is judged twice with A/B swapped. Judges favour whatever comes
     first; the swap cancels it, and the flip rate measures how much of the
     verdict is noise.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common
from agent.prompts import STYLE_SPECS
from agent.vlm import _extract_json

HERE = Path(__file__).resolve().parent

JUDGE_PROMPT = """You are grading two sets of video captions, A and B, written for the same clip.

Factual description of the clip:
{description}

Required styles:
{style_specs}

Set A:
{a}

Set B:
{b}

Score each set on two axes, 0-5 (integers):
- "accuracy": do the captions reflect the actual subjects, actions and setting of the description? Invented details lower this.
- "style": does each caption genuinely embody its style label? A "sarcastic" caption that is merely descriptive scores low; a "humorous_tech" caption with no technology reference scores low.

Then pick the better set overall. If they are genuinely equivalent, say "tie".

Respond with ONLY this JSON object:
{{"A": {{"accuracy": <0-5>, "style": <0-5>}}, "B": {{"accuracy": <0-5>, "style": <0-5>}}, "winner": "A" | "B" | "tie", "reason": "<one sentence>"}}"""


def description_of(prompt: str) -> str:
    """Recover the clip description from the production style_prompt."""
    m = re.search(r"Description:\n(.*?)\n\nWrite one caption per style\.", prompt, re.S)
    if not m:
        raise ValueError("style_prompt format changed — cannot extract description")
    return m.group(1).strip()


def normalized(completion: str) -> str:
    """Parse and re-serialize, so formatting cannot identify the arm."""
    parsed = _extract_json(completion)
    if not isinstance(parsed, dict):
        return ""
    return json.dumps({s: parsed.get(s, "") for s in common.STYLES}, indent=2, ensure_ascii=False)


def judge(description: str, first: str, second: str) -> dict:
    specs = "\n".join(f'- "{s}": {STYLE_SPECS[s]}' for s in common.STYLES)
    raw = common.chat(
        common.DESC_MODEL,
        JUDGE_PROMPT.format(description=description, style_specs=specs, a=first, b=second),
        max_tokens=600,
        temperature=0.0,  # a judge should be deterministic
    )
    verdict = _extract_json(raw)
    if not isinstance(verdict, dict) or "winner" not in verdict:
        raise ValueError(f"unparseable verdict: {raw[:200]}")
    return verdict


def judge_pair(task: tuple) -> dict:
    """Judge one val row twice: tuned-first, then opponent-first."""
    idx, description, tuned, opp, opp_name = task
    forward = judge(description, tuned, opp)   # A=tuned, B=opponent
    reverse = judge(description, opp, tuned)   # A=opponent, B=tuned

    def to_arms(v, a_is_tuned):
        t, o = ("A", "B") if a_is_tuned else ("B", "A")
        winner = {"tie": "tie", t: "tuned", o: opp_name}[v["winner"]]
        return {"tuned": v[t], opp_name: v[o], "winner": winner, "reason": v.get("reason", "")}

    return {"idx": idx, "forward": to_arms(forward, True), "reverse": to_arms(reverse, False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="judge eval_smoke.jsonl instead")
    ap.add_argument("--evals", type=Path, default=None,
                    help="path to any eval_*.jsonl (e.g. a quantized run's). "
                         "Judgements are written beside it.")
    ap.add_argument("--arm", default="tuned",
                    help="the `model` label in the eval file to judge (default: tuned)")
    ap.add_argument("--vs", choices=("base", "teacher"), default="base",
                    help="opponent: the un-adapted base model, or the teacher's own captions "
                         "(the ceiling — how much quality would serving Gemma locally cost?)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    opp_name = args.vs

    evals = args.evals or (HERE / "eval_results" /
                           ("eval_smoke.jsonl" if args.smoke else "eval_real.jsonl"))
    rows = [json.loads(line) for line in evals.read_text().splitlines()]

    val = [json.loads(line) for line in (common.DATA_DIR / "val.jsonl").read_text().splitlines()]
    by_arm = defaultdict(dict)
    for r in rows:
        by_arm[r["model"]][r["idx"]] = r["completion"]

    # `in`, not `by_arm[...]` — indexing a defaultdict creates the key it tests for
    if args.arm not in by_arm:
        sys.exit(f"no rows with model={args.arm!r} in {evals} "
                 f"(found: {sorted(by_arm) or 'nothing'})")

    tasks = []
    for idx in sorted(by_arm[args.arm]):
        # the teacher's captions are the assistant turn of the val row
        opponent_raw = (val[idx]["messages"][1]["content"] if opp_name == "teacher"
                        else by_arm["base"].get(idx, ""))
        tuned, opp = normalized(by_arm[args.arm][idx]), normalized(opponent_raw)
        if not tuned or not opp:
            print(f"row {idx}: unparseable completion in one arm — skipped")
            continue
        tasks.append((idx, description_of(val[idx]["messages"][0]["content"]), tuned, opp, opp_name))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(judge_pair, tasks))

    # judgements land beside their eval file, so a quantized run's results stay
    # with the quantized run. The arm under test is always labelled "tuned".
    stem = evals.stem.removeprefix("eval_")
    out = evals.parent / f"judged_{stem}_vs_{opp_name}.jsonl"
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(results)
    arms = ("tuned", opp_name)
    scores = {arm: {"accuracy": 0.0, "style": 0.0} for arm in arms}
    wins = defaultdict(int)
    flips = 0
    consistent = defaultdict(int)
    for r in results:
        for arm in arms:
            for axis in ("accuracy", "style"):
                # average the two orderings: cancels position bias
                scores[arm][axis] += (r["forward"][arm][axis] + r["reverse"][arm][axis]) / 2
        for direction in ("forward", "reverse"):
            wins[r[direction]["winner"]] += 1
        if r["forward"]["winner"] != r["reverse"]["winner"]:
            flips += 1
        else:
            consistent[r["forward"]["winner"]] += 1

    print(f"\njudged {n} val rows, {2 * n} judgements — tuned vs {opp_name} ({common.DESC_MODEL})\n")
    for arm in arms:
        a, s = scores[arm]["accuracy"] / n, scores[arm]["style"] / n
        print(f"  {arm:8}  accuracy {a:.2f}/5   style {s:.2f}/5")
    print(f"\n  wins: tuned {wins['tuned']}, {opp_name} {wins[opp_name]}, tie {wins['tie']}  (of {2 * n})")
    print(f"  order-flip rate: {flips}/{n} — the judge favours whatever it sees first; "
          f"only the verdicts that survive the swap mean anything")
    print(f"  verdicts surviving the swap: {n - flips}/{n} -> "
          f"tuned {consistent['tuned']}, {opp_name} {consistent[opp_name]}, tie {consistent['tie']}")
    print(f"\njudgements -> {out}")


if __name__ == "__main__":
    main()

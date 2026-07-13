"""Track 2 harness: /input/tasks.json -> /output/results.json, deadline-aware.

Reliability contract (missing style or malformed JSON scores zero, 10-minute kill):
1. A complete, valid results.json with fallback captions is written before any
   real work starts — a crash or timeout can never leave invalid/missing output.
2. Results are re-written atomically as each clip finishes.
3. The process force-exits 0 when the soft budget runs out, whatever is left.
"""

import json
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from agent import local_llm, prompts, video, vlm

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")
TIME_BUDGET = float(os.environ.get("TIME_BUDGET_SECONDS", "540"))  # harness kills at 600
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "3"))
DEFAULT_STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]

_START = time.monotonic()


def remaining():
    return TIME_BUDGET - (time.monotonic() - _START)


def write_results(order, results):
    payload = [{"task_id": tid, "captions": results[tid]} for tid in order]
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    tmp = OUTPUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUTPUT_PATH)


def process_task(task):
    tid = task.get("task_id")
    styles = task.get("styles") or DEFAULT_STYLES
    description = None
    captions = {}
    try:
        frames = video.sample_frames(task["video_url"], deadline=remaining)
        print("[%s] %d frames sampled (%.0fs left)" % (tid, len(frames), remaining()), flush=True)
        description = vlm.describe(frames, deadline=remaining)
        print("[%s] description: %s" % (tid, description[:120]), flush=True)
        captions = vlm.stylize(description, styles, deadline=remaining)
    except Exception:
        print("[%s] task failed, falling back" % tid, flush=True)
        traceback.print_exc()
    final = {}
    for s in styles:
        value = captions.get(s)
        if isinstance(value, str) and value.strip():
            final[s] = value.strip()
        else:
            final[s] = prompts.fallback_caption(s, description)
    return tid, final


def main():
    try:
        with open(INPUT_PATH) as f:
            tasks = json.load(f)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    order = []
    results = {}
    for t in tasks:
        tid = t.get("task_id")
        order.append(tid)
        styles = t.get("styles") or DEFAULT_STYLES
        results[tid] = {s: prompts.fallback_caption(s) for s in styles}
    write_results(order, results)  # valid output exists from second zero
    print("[main] %d tasks, %.0fs budget" % (len(tasks), remaining()), flush=True)

    # Only now: a crash inside the local model still leaves valid output behind.
    # Warming here also pays the one-time mmap page-in before any clip's timeout.
    if local_llm.ENABLED:
        local_llm.ensure_ready(deadline=remaining)

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    pending = {executor.submit(process_task, t) for t in tasks}
    while pending and remaining() > 5:
        done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
        for fut in done:
            try:
                tid, captions = fut.result()
                results[tid] = captions
                print("[main] %s done (%.0fs left)" % (tid, remaining()), flush=True)
            except Exception:
                traceback.print_exc()
        if done:
            write_results(order, results)

    write_results(order, results)
    if pending:
        print("[main] budget exhausted with %d tasks on fallbacks" % len(pending), flush=True)
    print("[main] results written to %s" % OUTPUT_PATH, flush=True)
    local_llm.shutdown()  # os._exit(0) skips atexit; don't orphan a 5 GB child
    # Straggler API/ffmpeg threads must not hold the container past the deadline:
    os._exit(0)


if __name__ == "__main__":
    main()

"""Optional local style stage: a llama-server subprocess serving the tuned GGUF.

Never fatal. Every failure condemns the local path for the rest of the run and
the caller falls back to the API — the same ladder as everything else in this
agent (local -> API -> template).

Four things here are load-bearing and none of them are defaults:

* `enable_thinking: false` — Gemma 4 has a thought channel and llama-server turns
  it ON. Thinking burns the whole token budget, `content` comes back EMPTY and
  `finish_reason=length`. Same failure Kimi K2.6 gives without reasoning_effort=none.
* `--swa-full` — Gemma uses sliding-window attention; llama.cpp refuses
  cross-request prefix reuse for SWA models unless the full cache is retained.
  Without it, prompt caching does nothing at all.
* `-np 1` — the server defaults to 4 slots and round-robins, so consecutive
  requests land on different KV caches and never share a prefix. One slot also
  makes our per-request timeouts mean something, since requests queue instead of
  fighting over the same 2 cores. Decode is compute-bound at low core counts;
  parallel slots would split cores without reducing total work.
* mmap (the default — do NOT pass --no-mmap/--mlock). Weights stay file-backed
  and reclaimable, so a tight container memory cap makes us SLOW rather than
  OOM-killed. Anonymous memory would kill the container instead.
"""

import os
import subprocess
import threading
import time

import requests

ENABLED = os.environ.get("LOCAL_STYLE", "0") == "1"
MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/models/gemma4-style-Q4_K_M.gguf")
SERVER_BIN = os.environ.get("LOCAL_SERVER_BIN", "/usr/local/bin/llama-server")
PORT = int(os.environ.get("LOCAL_PORT", "8080"))
CTX = os.environ.get("LOCAL_CTX", "4096")
THREADS = os.environ.get("LOCAL_THREADS", "")  # empty: derive from the cgroup quota
# Warm-up pays the mmap page-in cost once, up front, instead of inside the first
# clip's per-clip timeout (where it would look like a local-model failure).
WARMUP_TIMEOUT = float(os.environ.get("LOCAL_WARMUP_TIMEOUT", "120"))
# 461 prompt tokens of prefill on a slow 2-core box does not fit in 60 s.
REQUEST_TIMEOUT = float(os.environ.get("LOCAL_REQUEST_TIMEOUT", "120"))
# Below this much budget left, stop gambling on the slow path.
MIN_REMAINING = float(os.environ.get("LOCAL_MIN_REMAINING", "120"))
MAX_FAILURES = int(os.environ.get("LOCAL_MAX_FAILURES", "2"))

BASE_URL = "http://127.0.0.1:%d" % PORT

_start_lock = threading.Lock()  # guards spawn/warm-up
_req_lock = threading.Lock()    # -np 1 means one request at a time anyway
_proc = None
_ready = False
_condemned = not ENABLED
_failures = 0


def condemned():
    return _condemned


def _condemn(reason):
    global _condemned
    if not _condemned:
        _condemned = True
        print("[local] condemned: %s — falling back to the API" % reason, flush=True)


def note_failure(reason):
    """One strike. MAX_FAILURES strikes and the local path is out for good."""
    global _failures
    _failures += 1
    print("[local] failure %d/%d: %s" % (_failures, MAX_FAILURES, reason), flush=True)
    if _failures >= MAX_FAILURES:
        _condemn("too many failures")


def _quota_cpus():
    """How many CPUs this container may actually use.

    `docker run --cpus 2` is a CFS *quota*, not a core count: nproc still reports
    every host core. llama.cpp would spawn one thread per host core, and those
    OpenMP threads spin-wait against a 2-CPU budget — burning the quota on
    contention instead of matmuls. Read the cgroup and tell it the truth.
    """
    try:  # cgroup v2
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except Exception:
        pass
    try:  # cgroup v1
        quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if quota > 0:
            return max(1, quota // period)
    except Exception:
        pass
    return os.cpu_count() or 1


def _spawn():
    threads = THREADS or str(_quota_cpus())
    args = [
        SERVER_BIN,
        "-m", MODEL_PATH,
        "-c", CTX,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "-np", "1",        # one slot: prefix cache locality + meaningful timeouts
        "--swa-full",      # without this, Gemma's SWA disables prompt caching entirely
        "-t", threads,
    ]
    print("[local] starting (%s threads): %s" % (threads, " ".join(args)), flush=True)
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", threads)
    env.setdefault("OMP_WAIT_POLICY", "passive")  # don't spin away a CPU quota
    # stdout to /dev/null (token spam); stderr inherited so llama-server's own
    # diagnostics land in the container log next to ours.
    return subprocess.Popen(args, stdout=subprocess.DEVNULL, env=env)


def _wait_healthy(budget):
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        if _proc is not None and _proc.poll() is not None:
            raise RuntimeError("llama-server exited with code %s" % _proc.returncode)
        try:
            if requests.get(BASE_URL + "/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError("llama-server not healthy within %.0fs" % budget)


def ensure_ready(deadline=None):
    """Spawn + health-check + warm the mmap. Idempotent, never raises."""
    global _proc, _ready
    if _condemned or _ready:
        return _ready
    with _start_lock:
        if _condemned or _ready:
            return _ready
        budget = WARMUP_TIMEOUT
        if deadline is not None:
            budget = min(budget, max(deadline() - MIN_REMAINING, 0))
        if budget <= 0:
            _condemn("no budget left to warm up")
            return False
        if not os.path.exists(MODEL_PATH):
            _condemn("model not found at %s" % MODEL_PATH)
            return False
        started = time.monotonic()
        try:
            _proc = _spawn()
            _wait_healthy(budget)
            # First real generation faults the weights in from disk. Doing it here
            # keeps that one-time cost out of clip 1's timeout.
            chat("Reply with the single word: ok", deadline=None, max_tokens=8,
                 temperature=0.0, _warmup=True)
            _ready = True
            print("[local] ready in %.1fs" % (time.monotonic() - started), flush=True)
        except Exception as e:
            _condemn("warm-up failed (%s: %s)" % (type(e).__name__, e))
            shutdown()
        return _ready


def should_try(deadline=None):
    if _condemned:
        return False
    if not _ready and not ensure_ready(deadline):
        return False
    if deadline is not None and deadline() < MIN_REMAINING:
        print("[local] skipping: %.0fs left < %.0fs floor" % (deadline(), MIN_REMAINING), flush=True)
        return False
    return True


def chat(message, deadline=None, max_tokens=400, temperature=0.7, _warmup=False):
    """One user-turn completion. Raises on anything the caller shouldn't trust."""
    timeout = REQUEST_TIMEOUT
    if deadline is not None:
        timeout = min(REQUEST_TIMEOUT, max(deadline() - 5, 1))
    body = {
        "messages": [{"role": "user", "content": message}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": 42,
        "cache_prompt": True,
        # Gemma 4 thinks by default here; thinking returns EMPTY content.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with _req_lock:  # -np 1: serialize so a queued request can't outlive its timeout
        # A request that passed should_try() may have waited here while another
        # thread condemned the path. Don't spend a whole timeout proving it again.
        if _condemned and not _warmup:
            raise RuntimeError("local path condemned while this request was queued")
        resp = requests.post(BASE_URL + "/v1/chat/completions", json=body, timeout=timeout)
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    if choice.get("finish_reason") == "length":
        raise RuntimeError("local completion truncated (finish_reason=length)")
    content = (choice["message"].get("content") or "").strip()
    if not content:
        raise RuntimeError("local completion empty (thinking left on?)")
    if _warmup:
        print("[local] warm-up reply: %s" % content[:40].replace("\n", " "), flush=True)
    return content


def shutdown():
    """Best effort. os._exit(0) skips atexit, and the container teardown would
    reap the child anyway — but don't leave 5 GB resident if we exit early."""
    global _proc, _ready
    _ready = False
    if _proc is None:
        return
    try:
        _proc.terminate()
        _proc.wait(timeout=5)
    except Exception:
        try:
            _proc.kill()
        except Exception:
            pass
    _proc = None

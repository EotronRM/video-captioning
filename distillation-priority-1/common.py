"""Shared plumbing for the Phase A distillation scripts.

Bootstraps the project root onto sys.path so scripts can reuse the agent's
production code (agent.prompts.style_prompt, agent.vlm validation) — the
student must train on the exact input format and pass the exact validation
that production applies.
"""

import os
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]
DATA_DIR = Path(__file__).resolve().parent / "data"

# Text-only model for cheap description generation and judging.
DESC_MODEL = os.environ.get("DESC_MODEL", "accounts/fireworks/models/glm-5p2")
# The teacher whose style-captioning skill we distill.
TEACHER_MODEL = (
    os.environ.get("TEACHER_MODEL")
    or os.environ.get("VLM_MODEL")
    or "accounts/fireworks/models/kimi-k2p6"
)
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "none")

_client = None


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(
            base_url=os.environ.get(
                "VLM_BASE_URL", "https://api.fireworks.ai/inference/v1"
            ),
            api_key=os.environ["VLM_API_KEY"],
        )
    return _client


def chat(model, prompt, max_tokens=2000, temperature=0.7):
    """One user-turn completion with retries.

    Tolerates endpoints that reject reasoning_effort by dropping the param,
    and treats truncated/empty completions as failures (reasoning models leak
    thinking text into truncated content — never trust it).
    """
    send_reasoning = bool(REASONING_EFFORT)
    last = None
    for attempt in range(3):
        try:
            extra = {"reasoning_effort": REASONING_EFFORT} if send_reasoning else {}
            resp = get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=90,
                extra_body=extra,
            )
            if resp.choices[0].finish_reason == "length":
                raise RuntimeError("truncated (finish_reason=length)")
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                raise RuntimeError("empty content")
            return content
        except Exception as e:
            last = e
            if send_reasoning and "reasoning" in str(e).lower():
                send_reasoning = False  # endpoint rejects the param; drop it
            else:
                traceback.print_exc()
            time.sleep(2)
    raise last

"""OpenAI-compatible VLM client: one vision call per clip, one style call per clip.

Works unchanged against Fireworks, OpenRouter, or a self-hosted vLLM endpoint
(e.g. Qwen3-VL on an AMD MI300X) — only VLM_BASE_URL / VLM_MODEL change.
"""

import json
import os
import re
import time
import traceback

from agent import prompts

BASE_URL = os.environ.get("VLM_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.environ.get("VLM_API_KEY", "")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen/qwen3-vl-30b-a3b-instruct")
STYLE_MODEL = os.environ.get("STYLE_MODEL", "") or VLM_MODEL
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "25"))
# For reasoning models (e.g. Kimi K2.x on Fireworks): "none" disables thinking.
# Only sent when set — self-hosted endpoints may reject unknown params.
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "")
MOCK = os.environ.get("VLM_MOCK") == "1"

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI  # lazy: keeps VLM_MOCK plumbing tests dependency-free

        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY or "missing-key")
    return _client


def _strip_reasoning(text):
    """Drop inline thinking blocks some providers/models emit in content."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:  # opening tag truncated away
        text = text.split("</think>")[-1]
    return text.strip()


def _chat(model, messages, max_tokens, temperature, deadline):
    last_err = None
    for attempt in range(2):
        timeout = REQUEST_TIMEOUT
        if deadline is not None:
            timeout = min(REQUEST_TIMEOUT, max(deadline() - 5, 1))
        try:
            extra = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}
            resp = _get_client().chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature, timeout=timeout,
                extra_body=extra,
            )
            if resp.choices[0].finish_reason == "length":
                # Truncated generations from reasoning models leak thinking
                # text into content — never trust them.
                raise RuntimeError("completion truncated (finish_reason=length)")
            content = _strip_reasoning(resp.choices[0].message.content or "")
            if not content:
                # Reasoning models can burn the whole budget thinking; the
                # answer lands in `content` only if generation finished.
                raise RuntimeError(
                    "empty completion content (finish_reason=%s)"
                    % resp.choices[0].finish_reason
                )
            return content
        except Exception as e:
            last_err = e
            traceback.print_exc()
            time.sleep(1)
    raise last_err


def describe(frames_b64, deadline=None):
    """Vision stage: frames -> dense factual description (the accuracy dimension)."""
    if MOCK:
        return ("An orange kitten walks through green foliage in a sunlit garden, "
                "pausing to sniff a leaf while the camera follows at ground level.")
    content = [{"type": "text", "text": prompts.DESCRIBE_PROMPT}]
    for b in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + b},
        })
    text = _chat(VLM_MODEL, [{"role": "user", "content": content}],
                 max_tokens=1500, temperature=0.2, deadline=deadline)
    return text.strip()


def stylize(description, styles, deadline=None):
    """Style stage: one description -> {style: caption} (the style-match dimension)."""
    if MOCK:
        return {s: "[%s] mock caption about the kitten in the garden." % s for s in styles}
    message = prompts.style_prompt(description, styles)
    captions = {}
    for attempt in range(2):
        raw = _chat(STYLE_MODEL, [{"role": "user", "content": message}],
                    max_tokens=2000, temperature=0.7, deadline=deadline)
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            for s in styles:
                value = parsed.get(s)
                if isinstance(value, str) and _is_real_caption(value):
                    captions[s] = value.strip()
        if all(s in captions for s in styles):
            break
    return captions


def _is_real_caption(value):
    """Reject placeholder junk a confused model can echo back ('...', '', 'N/A')."""
    v = value.strip().strip(".…").strip()
    return len(v) >= 8 and v.lower() not in ("n/a", "todo", "caption")


def _extract_json(raw):
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None

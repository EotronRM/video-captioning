"""OpenAI-compatible VLM client: one vision call per clip, one style call per clip.

Works unchanged against Fireworks, OpenRouter, or a self-hosted vLLM endpoint
(e.g. Qwen3-VL on an AMD MI300X) — only VLM_BASE_URL / VLM_MODEL change.
"""

import json
import os
import re
import time
import traceback

from agent import local_llm, prompts

BASE_URL = os.environ.get("VLM_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.environ.get("VLM_API_KEY", "")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen/qwen3-vl-30b-a3b-instruct")
STYLE_MODEL = os.environ.get("STYLE_MODEL", "") or VLM_MODEL
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "25"))
# For reasoning models (e.g. Kimi K2.x on Fireworks): "none" disables thinking.
# Only sent when set — self-hosted endpoints may reject unknown params.
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "")
MOCK = os.environ.get("VLM_MOCK") == "1"
# "fallback": API first, local model only if the API fails (measured default).
# "primary" : local model first, API as the net.
LOCAL_ROLE = os.environ.get("LOCAL_ROLE", "fallback")

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


def _captions_from(raw, styles):
    """Pull the styles we can trust out of one raw completion."""
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        return {}
    found = {}
    for s in styles:
        value = parsed.get(s)
        if isinstance(value, str) and _is_real_caption(value):
            found[s] = value.strip()
    return found


def _try_local(message, styles, deadline):
    """Never raises: a local miss is a condemnation, not an error."""
    if not local_llm.should_try(deadline):
        return {}
    try:
        captions = _captions_from(local_llm.chat(message, deadline=deadline), styles)
        if all(s in captions for s in styles):
            return captions
        local_llm.note_failure("local returned %d/%d styles" % (len(captions), len(styles)))
    except Exception as e:
        local_llm.note_failure("%s: %s" % (type(e).__name__, e))
    return {}


def _try_api(message, styles, deadline):
    captions = {}
    for attempt in range(2):
        raw = _chat(STYLE_MODEL, [{"role": "user", "content": message}],
                    max_tokens=2000, temperature=0.7, deadline=deadline)
        captions.update(_captions_from(raw, styles))
        if all(s in captions for s in styles):
            break
    return captions


def stylize(description, styles, deadline=None):
    """Style stage: one description -> {style: caption} (the style-match dimension).

    LOCAL_ROLE=fallback (default): API first, local GGUF only if the API fails.
    The teacher beats the student on both axes (judged 10-3), and on 2 CPU cores
    the student is far slower — so it earns its place by surviving an outage, not
    by going first. A local caption still beats prompts.fallback_caption().

    LOCAL_ROLE=primary: local first, API as the net. Only sensible where the
    local model is fast enough to fit the per-clip budget; measure before trusting.
    """
    if MOCK:
        return {s: "[%s] mock caption about the kitten in the garden." % s for s in styles}
    message = prompts.style_prompt(description, styles)

    if LOCAL_ROLE == "primary":
        captions = _try_local(message, styles, deadline)
        if captions:
            return captions

    captions = {}
    try:
        captions = _try_api(message, styles, deadline)
    except Exception:
        traceback.print_exc()
    if all(s in captions for s in styles):
        return captions

    if LOCAL_ROLE == "fallback":
        local = _try_local(message, styles, deadline)
        if local:
            return local
    return captions  # partial: the caller templates whatever is missing


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

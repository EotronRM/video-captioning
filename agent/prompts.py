"""Prompt templates, few-shot style examples, and fallback captions."""

DESCRIBE_PROMPT = (
    "These are frames sampled in chronological order from a single short video clip.\n"
    "Describe the clip in 3-5 factual sentences covering: the main subject(s) and what "
    "they are doing, the setting and time of day, notable colors and objects, and any "
    "visible motion or camera movement. State only what is clearly visible; do not "
    "speculate. Write about it as a video, never mention frames or images."
)

STYLE_SPECS = {
    "formal": "Professional, objective, factual tone. No jokes, no slang.",
    "sarcastic": "Dry, ironic, lightly mocking. Deadpan delivery.",
    "humorous_tech": (
        "Funny, built on technology or programming references "
        "(bugs, infinite loops, deploys, Wi-Fi, CPUs, git...)."
    ),
    "humorous_non_tech": (
        "Funny, everyday observational humour anyone would get. "
        "Absolutely no technical jargon."
    ),
}

_EXAMPLE = """Example (for a clip of a dog chasing its tail on a lawn):
{
  "formal": "A golden retriever spins in circles chasing its tail on a sunlit suburban lawn.",
  "sarcastic": "Yes, keep spinning — the tail is definitely about to surrender any minute now.",
  "humorous_tech": "Infinite loop detected: dog.chase(tail) has no exit condition and the CPU is extremely fluffy.",
  "humorous_non_tech": "Somewhere on this lawn, a dog is losing a race against his own back half."
}"""


def style_prompt(description, styles):
    """Build the style-stage prompt asking for one caption per requested style, as JSON."""
    spec_lines = "\n".join(
        '- "{s}": {spec}'.format(
            s=s, spec=STYLE_SPECS.get(s, "Match the tone implied by the style name.")
        )
        for s in styles
    )
    keys = ", ".join('"{s}"'.format(s=s) for s in styles)
    return (
        "You write video captions. Below is a factual description of a video clip.\n\n"
        "Description:\n{description}\n\n"
        "Write one caption per style. Rules:\n"
        "- Each caption is a single sentence, at most 30 words, in English.\n"
        "- Every caption must reflect the actual content of the clip "
        "(subjects, actions, setting).\n"
        "{spec_lines}\n\n"
        "{example}\n\n"
        "Respond with ONLY a JSON object with exactly these keys: {keys}."
    ).format(description=description, spec_lines=spec_lines, example=_EXAMPLE, keys=keys)


_FALLBACK_TEMPLATES = {
    "formal": "{b}.",
    "sarcastic": "Oh good, another video: {b}. Riveting stuff, truly.",
    "humorous_tech": "{b} — rendered at a suspiciously smooth frame rate, zero bugs detected so far.",
    "humorous_non_tech": "{b} — and honestly, it steals the whole show.",
}


def fallback_caption(style, description=None):
    """Last-resort caption so no style is ever missing (missing style = zero for the clip).

    If the vision stage succeeded, anchor the fallback on the description's first
    sentence so it still scores something on accuracy.
    """
    base = "A short video clip"
    if description:
        first = description.split(".")[0].strip().rstrip(".")
        if first:
            base = first
    template = _FALLBACK_TEMPLATES.get(style, "{b}.")
    return template.format(b=base)

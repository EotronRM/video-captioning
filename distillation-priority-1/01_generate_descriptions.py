"""Phase A step 1: generate diverse scene descriptions.

Coverage mirrors the hackathon guide's hidden-set categories: nature, urban,
animals, people, sports, food, weather, technology. Diversity comes from a
seeded combinatorial grid (category x subject x time x camera x motion); the
text model only turns each combo into prose. Same --seed -> same combos, so
reruns are reproducible and 02's resume logic stays consistent.

Usage:
  uv run --env-file .env python distillation-priority-1/01_generate_descriptions.py --count 200
"""

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import common

CATEGORIES = {
    "nature": ["a mountain stream", "a dense forest canopy", "ocean waves on a rocky shore",
               "a field of wildflowers", "a desert dune at dusk"],
    "urban": ["a busy intersection", "a subway platform", "a night market street",
              "an old town square", "a construction site"],
    "animals": ["a kitten exploring a garden", "dogs playing in a park", "birds at a feeder",
                "horses in a paddock", "fish around a coral reef"],
    "people": ["an office worker at a desk", "a street musician performing", "children on a playground",
               "a barista making coffee", "two friends walking and talking"],
    "sports": ["a pickup basketball game", "a cyclist descending a hill", "a skateboarder attempting tricks",
               "runners at a race start", "a tennis rally"],
    "food": ["a chef plating a dish", "street food frying on a griddle", "someone kneading dough",
             "a pour-over coffee being brewed", "a sizzling barbecue grill"],
    "weather": ["heavy rain on a city street", "snow falling in a park", "fog rolling over hills",
                "a thunderstorm on the horizon", "wind bending palm trees"],
    "technology": ["a programmer debugging on dual monitors", "a robot arm assembling parts",
                   "a drone taking off", "a server room with blinking LEDs", "someone assembling a PC"],
}
TIMES = ["early morning", "midday", "golden hour", "night"]
CAMERA = ["a static tripod shot", "a slow pan", "handheld follow footage",
          "an aerial drone shot", "a timelapse", "a low-angle close-up"]
MOTION = ["very little motion", "moderate movement", "fast continuous action"]

PROMPT = """Write a factual description of a short video clip (30 seconds to 2 minutes), \
exactly as a vision model would describe it after watching sampled frames.

Scene brief:
- Category: {category}
- Subject: {subject}
- Time: {time}
- Camera: {camera}
- Overall motion: {motion}

Rules:
- 3 to 5 sentences, 60-120 words.
- Cover: main subject(s) and what they are doing, the setting, notable colors \
and objects, visible motion and camera movement.
- Only concrete visible facts; no speculation, no opinions, no mention of frames or images.
Respond with the description only."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cats = list(CATEGORIES)
    combos = []
    for i in range(args.count):
        cat = cats[i % len(cats)]  # even category coverage
        combos.append({
            "id": "d%04d" % i,
            "category": cat,
            "subject": rng.choice(CATEGORIES[cat]),
            "time": rng.choice(TIMES),
            "camera": rng.choice(CAMERA),
            "motion": rng.choice(MOTION),
        })

    def gen(combo):
        try:
            text = common.chat(
                common.DESC_MODEL,
                PROMPT.format(category=combo["category"], subject=combo["subject"],
                              time=combo["time"], camera=combo["camera"],
                              motion=combo["motion"]),
                temperature=0.9,
            )
            if len(text.split()) < 30:  # refused / degenerate output
                raise ValueError("description too short")
            return {**combo, "description": text}
        except Exception as e:
            print("[%s] failed: %s" % (combo["id"], str(e)[:120]), flush=True)
            return None

    common.DATA_DIR.mkdir(exist_ok=True)
    out_path = common.DATA_DIR / "descriptions.jsonl"
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex, open(out_path, "w") as f:
        for rec in ex.map(gen, combos):
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done += 1
                if done % 20 == 0:
                    print("%d/%d written" % (done, len(combos)), flush=True)
    print("wrote %d/%d descriptions -> %s" % (done, len(combos), out_path))
    if done < args.count * 0.9:
        sys.exit(1)  # too many failures to be a usable dataset


if __name__ == "__main__":
    main()

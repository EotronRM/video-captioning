"""Video frame sampling: ffmpeg keyframe seeks, remote-first with full-download fallback."""

import base64
import os
import subprocess
import tempfile
import traceback

FRAMES_PER_CLIP = int(os.environ.get("FRAMES_PER_CLIP", "8"))
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", "512"))
FFMPEG_TIMEOUT = float(os.environ.get("FFMPEG_TIMEOUT", "20"))
DOWNLOAD_TIMEOUT = float(os.environ.get("DOWNLOAD_TIMEOUT", "120"))


def probe_duration(src):
    """Clip duration in seconds. `src` may be a local path or an http(s) URL."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", src],
        capture_output=True, text=True, timeout=FFMPEG_TIMEOUT,
    )
    if out.returncode != 0:
        raise RuntimeError("ffprobe failed: " + out.stderr.strip()[:300])
    return float(out.stdout.strip())


def _grab_frame(src, ts, width, path):
    """Grab one downscaled JPEG at timestamp `ts`. -ss before -i = fast keyframe seek,
    so over HTTP ffmpeg range-requests only the bytes it needs instead of the whole file."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", "%.2f" % ts, "-i", src,
         "-frames:v", "1", "-vf", "scale=%d:-2" % width, "-q:v", "3", "-y", path],
        capture_output=True, text=True, timeout=FFMPEG_TIMEOUT,
    )
    if out.returncode != 0 or not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError("ffmpeg frame grab failed: " + out.stderr.strip()[:300])


def _sample(src, n, width, deadline):
    duration = probe_duration(src)
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n):
            if deadline is not None and deadline() < 15:
                break  # out of time budget; caption from what we have
            ts = duration * (i + 0.5) / n
            path = os.path.join(tmp, "f%02d.jpg" % i)
            try:
                _grab_frame(src, ts, width, path)
            except Exception:
                continue  # tolerate a bad seek point, keep the rest
            with open(path, "rb") as f:
                frames.append(base64.b64encode(f.read()).decode())
    if not frames:
        raise RuntimeError("no frames extracted from " + src)
    return frames


def _download(url, path, deadline):
    import requests  # lazy: keeps VLM_MOCK plumbing tests dependency-free

    with requests.get(url, stream=True, timeout=(10, DOWNLOAD_TIMEOUT)) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if deadline is not None and deadline() < 20:
                    raise RuntimeError("download aborted: time budget nearly exhausted")
                f.write(chunk)


def sample_frames(url, n=FRAMES_PER_CLIP, width=FRAME_WIDTH, deadline=None):
    """Return up to `n` base64 JPEGs sampled uniformly across the clip.

    Tries seeking directly against the URL first (downloads ~MBs instead of the
    full UHD file); falls back to a full download if the remote seeks fail.
    """
    if os.environ.get("VLM_MOCK") == "1":
        return []  # plumbing-test mode: no ffmpeg, no network

    try:
        return _sample(url, n, width, deadline)
    except Exception:
        print("[video] remote seek failed for %s, falling back to download" % url, flush=True)
        traceback.print_exc()

    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "clip.mp4")
        _download(url, local, deadline)
        return _sample(local, n, width, deadline)

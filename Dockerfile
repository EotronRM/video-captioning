FROM python:3.12-slim

# ffmpeg for frame extraction; uv for reproducible dependency installs
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY agent ./agent

ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["python", "-m", "agent.main"]

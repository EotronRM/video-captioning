#!/usr/bin/env bash
# Build the container and run it against the three example clips.
# Requires Docker and a .env file (copy .env.example and fill in your API key).
set -euo pipefail
cd "$(dirname "$0")/.."

docker build -t track2-agent .
mkdir -p out
# -e flags come after --env-file so container paths win over any local
# INPUT_PATH/OUTPUT_PATH overrides in .env
docker run --rm \
  -v "$PWD/sample_input:/input:ro" \
  -v "$PWD/out:/output" \
  --env-file .env \
  -e INPUT_PATH=/input/tasks.json \
  -e OUTPUT_PATH=/output/results.json \
  track2-agent

echo "--- results ---"
cat out/results.json

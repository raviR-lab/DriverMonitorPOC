#!/usr/bin/env bash
#
# One-command demo launcher for the Edge AI DMS.
# Phase 1: runs main.py on videos/dataset.mp4 with the SSE UI enabled.
#
# Usage:
#   bash scripts/run_demo.sh                # default: dataset.mp4, no save
#   bash scripts/run_demo.sh --save         # also write annotated output video

set -e
cd "$(dirname "$0")/.."

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [ -d "venv" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
else
  echo "ERROR: no virtualenv found (.venv or venv)"
  exit 1
fi

if [ ! -f "videos/dataset.mp4" ]; then
  echo "ERROR: videos/dataset.mp4 not found"
  exit 1
fi

HOST_IP=$(ipconfig getifaddr en0 2>/dev/null || echo "127.0.0.1")
echo "=================================================="
echo " Edge AI DMS — Phase 1 demo"
echo " UI:    http://${HOST_IP}:5050"
echo " Local: http://127.0.0.1:5050"
echo "=================================================="

python main.py --video videos/dataset.mp4 "$@"

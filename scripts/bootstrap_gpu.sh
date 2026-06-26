#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3.12}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python 3.12 was not found. Set PYTHON=/path/to/python3.12 or install Python 3.12." >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[gpu]"

echo
echo "Environment ready."
echo "Activate it with: source .venv/bin/activate"
echo "Smoke test: transcribe-jp /path/to/video.mp4 --dry-run"

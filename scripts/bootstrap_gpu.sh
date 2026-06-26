#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "No python3 found. Set PYTHON=/path/to/python3 (3.10 or newer)." >&2
  exit 1
fi

# qwen-asr supports Python 3.9-3.13; this project targets 3.10+.
if ! "${PYTHON_BIN}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Python 3.10 or newer is required. Found: $(${PYTHON_BIN} --version 2>&1)" >&2
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

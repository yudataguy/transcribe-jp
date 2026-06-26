#!/usr/bin/env bash
# Batch-transcribe many audio/video files with transcribe-jp on the GPU machine.
# Accepts files and/or directories (scanned recursively). Files whose .txt
# output already exists are skipped, so the batch is resumable. Any arguments
# after `--` are forwarded verbatim to transcribe-jp.
set -euo pipefail

OUTDIR="outputs"
FORCE=0

MEDIA_EXTS="mp4 mkv mov avi webm m4v flv wmv ts mpg mpeg mp3 m4a aac wav flac ogg oga opus"

usage() {
  cat <<'EOF'
Usage: batch_transcribe.sh [-o OUTDIR] [-F] INPUT... [-- TRANSCRIBE_JP_ARGS...]

Run transcribe-jp on every INPUT (file or directory) into OUTDIR.

Options:
  -o OUTDIR   Output directory (default: outputs).
  -F          Re-transcribe even if OUTDIR/<name>.txt already exists.
  -h          Show this help.

Anything after `--` is passed straight to transcribe-jp.

Examples:
  batch_transcribe.sh audio/
  batch_transcribe.sh audio/ -- --max-batch-size 4 --forced-aligner
  batch_transcribe.sh -o out a.opus b.opus -- --attn-impl flash_attention_2
EOF
}

while getopts ":o:Fh" opt; do
  case "$opt" in
    o) OUTDIR="$OPTARG" ;;
    F) FORCE=1 ;;
    h) usage; exit 0 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage >&2; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

# Split remaining args into inputs (before `--`) and passthrough (after `--`).
raw_inputs=()
passthrough=()
seen_dd=0
for arg in "$@"; do
  if [ "$seen_dd" -eq 0 ] && [ "$arg" = "--" ]; then
    seen_dd=1
    continue
  fi
  if [ "$seen_dd" -eq 1 ]; then
    passthrough+=("$arg")
  else
    raw_inputs+=("$arg")
  fi
done

if [ "${#raw_inputs[@]}" -eq 0 ]; then
  usage >&2
  exit 1
fi

if ! command -v transcribe-jp >/dev/null 2>&1; then
  echo "transcribe-jp not found. Activate the venv (source .venv/bin/activate)." >&2
  exit 1
fi

find_expr=()
first=1
for ext in $MEDIA_EXTS; do
  if [ "$first" -eq 1 ]; then first=0; else find_expr+=(-o); fi
  find_expr+=(-iname "*.$ext")
done

inputs=()
for arg in "${raw_inputs[@]}"; do
  if [ -d "$arg" ]; then
    while IFS= read -r -d '' f; do
      inputs+=("$f")
    done < <(find "$arg" -type f \( "${find_expr[@]}" \) -print0 | sort -z)
  elif [ -f "$arg" ]; then
    inputs+=("$arg")
  else
    echo "Skipping (not found): $arg" >&2
  fi
done

if [ "${#inputs[@]}" -eq 0 ]; then
  echo "No media files found." >&2
  exit 1
fi

mkdir -p "$OUTDIR"
ok=0
skipped=0
failed=0
total="${#inputs[@]}"
n=0

for f in "${inputs[@]}"; do
  n=$((n + 1))
  stem="$(basename "${f%.*}")"
  if [ "$FORCE" -eq 0 ] && [ -f "$OUTDIR/$stem.txt" ]; then
    echo "[$n/$total] skip (done): $stem"
    skipped=$((skipped + 1))
    continue
  fi
  echo "[$n/$total] transcribing: $f"
  if transcribe-jp "$f" --output-dir "$OUTDIR" ${passthrough[@]+"${passthrough[@]}"}; then
    ok=$((ok + 1))
  else
    echo "FAILED: $f" >&2
    failed=$((failed + 1))
  fi
done

echo "Done. transcribed=$ok skipped=$skipped failed=$failed -> $OUTDIR/"
[ "$failed" -eq 0 ]

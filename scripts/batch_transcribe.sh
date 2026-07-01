#!/usr/bin/env bash
# Batch-transcribe many audio/video files with transcribe-jp on the GPU machine.
# Accepts files and/or directories (scanned recursively). Files whose .txt
# output already exists are skipped, so the batch is resumable. Any arguments
# after `--` are forwarded verbatim to transcribe-jp.
set -euo pipefail

OUTDIR="outputs"
FORCE=0

MEDIA_EXTS="mp4 mkv mov avi webm m4v flv wmv ts mpg mpeg mp3 m4a aac wav flac ogg oga opus"
PROGRESS_WIDTH=28
child_pid=""

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

repeat_char() {
  local char="$1"
  local count="$2"
  local out=""
  if [ "$count" -gt 0 ]; then
    printf -v out "%${count}s" ""
    out="${out// /$char}"
  fi
  printf "%s" "$out"
}

short_name() {
  local name="$1"
  local max="${2:-44}"
  if [ "${#name}" -le "$max" ]; then
    printf "%s" "$name"
  else
    printf "...%s" "${name: -$((max - 3))}"
  fi
}

format_elapsed() {
  local seconds="$1"
  printf "%02d:%02d" "$((seconds / 60))" "$((seconds % 60))"
}

render_progress() {
  local current="$1"
  local total="$2"
  local label="$3"
  local width="${4:-$PROGRESS_WIDTH}"
  local percent=100
  local filled="$width"

  if [ "$total" -gt 0 ]; then
    if [ "$current" -gt "$total" ]; then
      current="$total"
    fi
    percent=$((current * 100 / total))
    filled=$((current * width / total))
  fi

  local empty=$((width - filled))
  local bar
  bar="$(repeat_char "#" "$filled")$(repeat_char "." "$empty")"

  if [ -t 2 ]; then
    printf "\r%-46s [%s] %3d%%" "$label" "$bar" "$percent" >&2
  else
    printf "%s [%s] %3d%%\n" "$label" "$bar" "$percent" >&2
  fi
}

finish_progress() {
  render_progress "$1" "$2" "$3"
  printf "\n" >&2
}

render_activity() {
  local label="$1"
  local elapsed="$2"
  local tick="$3"
  local width="${4:-$PROGRESS_WIDTH}"
  local filled=$((tick % (width + 1)))
  local empty=$((width - filled))
  local bar
  bar="$(repeat_char "#" "$filled")$(repeat_char "." "$empty")"

  if [ -t 2 ]; then
    printf "\r%-46s [%s] elapsed %s" "$label" "$bar" "$(format_elapsed "$elapsed")" >&2
  else
    printf "%s [%s] elapsed %s\n" "$label" "$bar" "$(format_elapsed "$elapsed")" >&2
  fi
}

finish_activity() {
  local label="$1"
  local elapsed="$2"
  local status="$3"
  local suffix="done"
  if [ "$status" -ne 0 ]; then
    suffix="failed"
  fi
  local bar
  bar="$(repeat_char "#" "$PROGRESS_WIDTH")"
  if [ -t 2 ]; then
    printf "\r%-46s [%s] elapsed %s %s\n" "$label" "$bar" "$(format_elapsed "$elapsed")" "$suffix" >&2
  else
    printf "%s [%s] elapsed %s %s\n" "$label" "$bar" "$(format_elapsed "$elapsed")" "$suffix" >&2
  fi
}

cleanup_child() {
  if [ -n "$child_pid" ] && kill -0 "$child_pid" 2>/dev/null; then
    kill "$child_pid" 2>/dev/null || true
  fi
}

trap 'cleanup_child; exit 130' INT
trap 'cleanup_child; exit 143' TERM

run_with_activity() {
  local label="$1"
  shift
  local start
  local now
  local elapsed
  local tick=0
  local status

  start="$(date +%s)"
  "$@" &
  child_pid="$!"

  while kill -0 "$child_pid" 2>/dev/null; do
    now="$(date +%s)"
    elapsed=$((now - start))
    render_activity "$label" "$elapsed" "$tick"
    tick=$((tick + 1))
    sleep 1
  done

  set +e
  wait "$child_pid"
  status="$?"
  set -e
  child_pid=""

  now="$(date +%s)"
  elapsed=$((now - start))
  finish_activity "$label" "$elapsed" "$status"
  return "$status"
}

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
  batch_label="Overall $n/$total"
  if [ "$FORCE" -eq 0 ] && [ -f "$OUTDIR/$stem.txt" ]; then
    finish_progress "$n" "$total" "$batch_label"
    echo "skip (done): $stem"
    skipped=$((skipped + 1))
    continue
  fi
  finish_progress "$((n - 1))" "$total" "$batch_label"
  echo "transcribing: $f"
  cmd=(transcribe-jp "$f" --output-dir "$OUTDIR")
  cmd+=("${passthrough[@]}")
  if run_with_activity "File $(short_name "$(basename "$f")")" "${cmd[@]}"; then
    ok=$((ok + 1))
  else
    echo "FAILED: $f" >&2
    failed=$((failed + 1))
  fi
  finish_progress "$n" "$total" "$batch_label"
done

echo "Done. transcribed=$ok skipped=$skipped failed=$failed -> $OUTDIR/"
[ "$failed" -eq 0 ]

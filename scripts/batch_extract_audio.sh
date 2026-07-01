#!/usr/bin/env bash
# Batch-extract model-ready audio from many media files before uploading.
# Accepts files and/or directories (directories are scanned recursively for
# common media extensions). Already-extracted outputs are skipped so the batch
# is resumable. Run locally on your laptop.
set -euo pipefail

FORMAT="opus"
BITRATE=""
OUTDIR="audio"
FORCE=0

MEDIA_EXTS="mp4 mkv mov avi webm m4v flv wmv ts mpg mpeg mp3 m4a aac wav flac ogg oga opus"
PROGRESS_WIDTH=28

usage() {
  cat <<'EOF'
Usage: batch_extract_audio.sh [-f opus|mp3|wav] [-b BITRATE] [-o OUTDIR] [-F] INPUT...

Extract 16 kHz mono audio from every INPUT (file or directory) into OUTDIR.

Options:
  -f FORMAT   opus (default), mp3, or wav.
  -b BITRATE  e.g. 16k, 24k, 32k (ignored for wav).
  -o OUTDIR   Output directory (default: audio).
  -F          Force re-extraction even if the output already exists.
  -h          Show this help.

Examples:
  batch_extract_audio.sh videos/                 # every media file in videos/ -> audio/*.opus
  batch_extract_audio.sh -o out -f mp3 a.mp4 b.mkv
EOF
}

while getopts ":f:b:o:Fh" opt; do
  case "$opt" in
    f) FORMAT="$OPTARG" ;;
    b) BITRATE="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    F) FORCE=1 ;;
    h) usage; exit 0 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
    \?) echo "Unknown option: -$OPTARG" >&2; usage >&2; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

if [ "$#" -lt 1 ]; then
  usage >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg was not found. Install it first (e.g. 'brew install ffmpeg')." >&2
  exit 1
fi

case "$FORMAT" in
  opus) CODEC="libopus"; EXT="opus"; DEFAULT_BITRATE="16k" ;;
  mp3) CODEC="libmp3lame"; EXT="mp3"; DEFAULT_BITRATE="32k" ;;
  wav) CODEC="pcm_s16le"; EXT="wav"; DEFAULT_BITRATE="" ;;
  *) echo "Unsupported format: $FORMAT (use opus, mp3, or wav)." >&2; exit 1 ;;
esac

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

media_duration_us() {
  local input="$1"
  local duration=""
  if command -v ffprobe >/dev/null 2>&1; then
    duration="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$input" 2>/dev/null || true)"
  fi
  awk -v duration="$duration" 'BEGIN { if (duration > 0) printf "%.0f", duration * 1000000; else print 0 }'
}

extract_with_progress() {
  local input="$1"
  local output="$2"
  local file_label="$3"
  local duration_us
  local err_log
  local key
  local value
  local status

  duration_us="$(media_duration_us "$input")"
  err_log="$(mktemp "${TMPDIR:-/tmp}/batch_extract_audio.XXXXXX")"

  local ffmpeg_args=(-hide_banner -loglevel error -y -i "$input" -vn -ac 1 -ar 16000 -c:a "$CODEC")
  if [ "$FORMAT" != "wav" ]; then
    ffmpeg_args+=(-b:a "${BITRATE:-$DEFAULT_BITRATE}")
  fi
  ffmpeg_args+=(-progress pipe:1 -nostats "$output")

  if [ "$duration_us" -gt 0 ]; then
    render_progress 0 "$duration_us" "$file_label"
  else
    render_progress 0 1 "$file_label"
  fi
  set +e
  ffmpeg "${ffmpeg_args[@]}" 2>"$err_log" | while IFS='=' read -r key value; do
    if [ "$key" = "out_time_ms" ] && [[ "$value" =~ ^[0-9]+$ ]] && [ "$duration_us" -gt 0 ] && [ "$value" -lt "$duration_us" ]; then
      render_progress "$value" "$duration_us" "$file_label"
    fi
  done
  status="${PIPESTATUS[0]}"
  set -e

  if [ "$status" -eq 0 ]; then
    if [ "$duration_us" -gt 0 ]; then
      finish_progress "$duration_us" "$duration_us" "$file_label"
    else
      finish_progress 1 1 "$file_label"
    fi
  else
    printf "\n" >&2
    cat "$err_log" >&2
  fi
  rm -f "$err_log"
  return "$status"
}

# Build a reusable find expression: \( -iname *.ext1 -o -iname *.ext2 ... \)
find_expr=()
first=1
for ext in $MEDIA_EXTS; do
  if [ "$first" -eq 1 ]; then first=0; else find_expr+=(-o); fi
  find_expr+=(-iname "*.$ext")
done

# Expand inputs (files kept as-is, directories scanned) into a flat list.
inputs=()
for arg in "$@"; do
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
  out="$OUTDIR/$stem.$EXT"
  batch_label="Overall $n/$total"
  if [ "$FORCE" -eq 0 ] && [ -f "$out" ]; then
    finish_progress "$n" "$total" "$batch_label"
    echo "skip (exists): $out"
    skipped=$((skipped + 1))
    continue
  fi
  finish_progress "$((n - 1))" "$total" "$batch_label"
  echo "extracting: $f -> $out"
  if extract_with_progress "$f" "$out" "File $(short_name "$(basename "$f")")"; then
    ok=$((ok + 1))
  else
    echo "FAILED: $f" >&2
    failed=$((failed + 1))
  fi
  finish_progress "$n" "$total" "$batch_label"
done

echo "Done. extracted=$ok skipped=$skipped failed=$failed -> $OUTDIR/"
[ "$failed" -eq 0 ]

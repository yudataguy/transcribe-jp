#!/usr/bin/env bash
# Batch-extract model-ready audio from many media files before uploading.
# Wraps scripts/extract_audio.sh; accepts files and/or directories (directories
# are scanned recursively for common media extensions). Already-extracted
# outputs are skipped so the batch is resumable. Run locally on your laptop.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACT="$SCRIPT_DIR/extract_audio.sh"

FORMAT="opus"
BITRATE=""
OUTDIR="audio"
FORCE=0

MEDIA_EXTS="mp4 mkv mov avi webm m4v flv wmv ts mpg mpeg mp3 m4a aac wav flac ogg oga opus"

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

if [ ! -x "$EXTRACT" ]; then
  echo "Cannot find extract_audio.sh next to this script ($EXTRACT)." >&2
  exit 1
fi

case "$FORMAT" in
  opus) EXT="opus" ;;
  mp3) EXT="mp3" ;;
  wav) EXT="wav" ;;
  *) echo "Unsupported format: $FORMAT (use opus, mp3, or wav)." >&2; exit 1 ;;
esac

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

for f in "${inputs[@]}"; do
  stem="$(basename "${f%.*}")"
  out="$OUTDIR/$stem.$EXT"
  if [ "$FORCE" -eq 0 ] && [ -f "$out" ]; then
    echo "skip (exists): $out"
    skipped=$((skipped + 1))
    continue
  fi
  args=(-f "$FORMAT")
  if [ -n "$BITRATE" ]; then
    args+=(-b "$BITRATE")
  fi
  if "$EXTRACT" "${args[@]}" "$f" "$out"; then
    ok=$((ok + 1))
  else
    echo "FAILED: $f" >&2
    failed=$((failed + 1))
  fi
done

echo "Done. extracted=$ok skipped=$skipped failed=$failed -> $OUTDIR/"
[ "$failed" -eq 0 ]

#!/usr/bin/env bash
# Extract model-ready audio (16 kHz mono) from a media file before uploading.
# Defaults to Opus 16 kbps, which is ~100-300x smaller than a source video and
# transparent to Qwen3-ASR (it only ever consumes 16 kHz mono audio).
set -euo pipefail

FORMAT="opus"
BITRATE=""

usage() {
  cat <<'EOF'
Usage: extract_audio.sh [-f opus|mp3|wav] [-b BITRATE] INPUT [OUTPUT]

Extract 16 kHz mono audio suitable for transcribe-jp, minimizing upload size.

Options:
  -f FORMAT   Output codec/container: opus (default), mp3, or wav.
  -b BITRATE  Audio bitrate, e.g. 16k, 24k, 32k. Ignored for wav.
              Defaults: opus=16k, mp3=32k.
  -h          Show this help.

Examples:
  extract_audio.sh video.mp4                  # -> video.opus (16k)
  extract_audio.sh -f mp3 video.mp4           # -> video.mp3 (32k)
  extract_audio.sh -b 24k video.mp4 out.opus  # -> out.opus (24k)
EOF
}

while getopts ":f:b:h" opt; do
  case "$opt" in
    f) FORMAT="$OPTARG" ;;
    b) BITRATE="$OPTARG" ;;
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

INPUT="$1"
OUTPUT="${2:-}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg was not found. Install it first (e.g. 'brew install ffmpeg')." >&2
  exit 1
fi

if [ ! -f "$INPUT" ]; then
  echo "Input file not found: $INPUT" >&2
  exit 1
fi

case "$FORMAT" in
  opus) CODEC="libopus"; EXT="opus"; DEFAULT_BITRATE="16k" ;;
  mp3) CODEC="libmp3lame"; EXT="mp3"; DEFAULT_BITRATE="32k" ;;
  wav) CODEC="pcm_s16le"; EXT="wav"; DEFAULT_BITRATE="" ;;
  *) echo "Unsupported format: $FORMAT (use opus, mp3, or wav)." >&2; exit 1 ;;
esac

if [ -z "$OUTPUT" ]; then
  OUTPUT="${INPUT%.*}.${EXT}"
fi

if [ "$OUTPUT" = "$INPUT" ]; then
  echo "Output would overwrite the input ($INPUT). Pass a different OUTPUT." >&2
  exit 1
fi

# -vn drops video; -ac 1 downmixes to mono; -ar 16000 resamples to 16 kHz.
ffmpeg_args=(-hide_banner -loglevel error -y -i "$INPUT" -vn -ac 1 -ar 16000 -c:a "$CODEC")
if [ "$FORMAT" != "wav" ]; then
  ffmpeg_args+=(-b:a "${BITRATE:-$DEFAULT_BITRATE}")
fi
ffmpeg_args+=("$OUTPUT")

ffmpeg "${ffmpeg_args[@]}"

size="$(du -h "$OUTPUT" | cut -f1)"
echo "Wrote ${OUTPUT} (${size})."
echo "Upload it, then run: transcribe-jp ${OUTPUT##*/} --output-dir outputs"

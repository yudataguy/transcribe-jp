from __future__ import annotations

import argparse
from pathlib import Path

from transcribe_jp.backends import transcribe_audio
from transcribe_jp.config import (
    DEFAULT_FORCED_ALIGNER,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_MODEL,
    DEFAULT_WINDOW_SECONDS,
    TranscriptionConfig,
)
from transcribe_jp.media import build_audio_path, build_ffmpeg_command, build_output_base, extract_audio
from transcribe_jp.transcript import (
    SRT_MAX_CHARS,
    SRT_MAX_DURATION,
    SRT_MAX_GAP,
    SRT_LINE_WIDTH,
    SrtOptions,
    write_transcript,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe-jp",
        description="Transcribe Japanese video or audio with Qwen3-ASR.",
    )
    parser.add_argument("media", type=Path, help="Path to a video or audio file.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument(
        "--backend",
        choices=("qwen-cli", "transformers"),
        default="transformers",
        help="Inference runner to use on the GPU machine.",
    )
    parser.add_argument(
        "--command-template",
        help=(
            "Command used by the qwen-cli backend. Available placeholders: "
            "{model}, {language}, {audio}."
        ),
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
        help=(
            "Audio chunks decoded in parallel by the transformers backend. "
            "Lower this (e.g. 1) if you hit CUDA out-of-memory; raise it for "
            "faster throughput on larger GPUs."
        ),
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help=(
            "Target length of each silence-aligned audio window for the "
            "transformers backend. Smaller windows give finer progress "
            "granularity and lower per-window memory."
        ),
    )
    parser.add_argument(
        "--forced-aligner",
        nargs="?",
        const=DEFAULT_FORCED_ALIGNER,
        default=None,
        metavar="MODEL",
        help=(
            "Enable word/phrase-level .srt timestamps via a forced-aligner "
            "model (loads a second ~0.6B model). Pass the flag alone to use "
            f"{DEFAULT_FORCED_ALIGNER}, or supply a custom model id. Omit for "
            "coarse per-window cues."
        ),
    )
    srt_group = parser.add_argument_group("subtitle (.srt) grouping")
    srt_group.add_argument(
        "--srt-max-chars",
        type=int,
        default=SRT_MAX_CHARS,
        help="Max characters before a cue is broken (default: %(default)s).",
    )
    srt_group.add_argument(
        "--srt-max-duration",
        type=float,
        default=SRT_MAX_DURATION,
        help="Max on-screen seconds per cue (default: %(default)s).",
    )
    srt_group.add_argument(
        "--srt-max-gap",
        type=float,
        default=SRT_MAX_GAP,
        help="Pause in seconds that forces a new cue (default: %(default)s).",
    )
    srt_group.add_argument(
        "--srt-line-width",
        type=int,
        default=SRT_LINE_WIDTH,
        help="Wrap cue text to this many characters per line (default: %(default)s).",
    )
    parser.add_argument("--keep-audio", action="store_true", help="Keep extracted WAV audio.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = TranscriptionConfig(
        model=args.model,
        language=args.language,
        output_dir=args.output_dir,
        backend=args.backend,
        keep_audio=args.keep_audio,
        command_template=args.command_template,
        max_batch_size=args.max_batch_size,
        window_seconds=args.window_seconds,
        forced_aligner=args.forced_aligner,
    )

    srt_options = SrtOptions(
        max_chars=args.srt_max_chars,
        max_duration=args.srt_max_duration,
        max_gap=args.srt_max_gap,
        line_width=args.srt_line_width,
    )

    media_path = args.media.expanduser()
    audio_path = build_audio_path(media_path, config.output_dir)
    output_base = build_output_base(media_path, config.output_dir)

    if args.dry_run:
        _print_dry_run(media_path, audio_path, output_base, config)
        return 0

    if not media_path.exists():
        parser.error(f"Media file does not exist: {media_path}")

    extract_audio(media_path, audio_path)
    transcript = transcribe_audio(audio_path, config)
    written = write_transcript(transcript, output_base, srt_options)

    if not config.keep_audio:
        audio_path.unlink(missing_ok=True)

    for kind, path in written.items():
        print(f"{kind}: {path}")
    return 0


def _print_dry_run(
    media_path: Path,
    audio_path: Path,
    output_base: Path,
    config: TranscriptionConfig,
) -> None:
    print("Transcription plan")
    print(f"media: {media_path}")
    print(f"model: {config.model}")
    print(f"language: {config.language}")
    print(f"backend: {config.backend}")
    print(f"max_batch_size: {config.max_batch_size}")
    print(f"window_seconds: {config.window_seconds}")
    print(f"forced_aligner: {config.forced_aligner or '(disabled)'}")
    print(f"audio: {audio_path}")
    print("ffmpeg:")
    print("  " + " ".join(build_ffmpeg_command(media_path, audio_path)))
    print("outputs:")
    print(f"  {output_base.with_suffix('.txt')}")
    print(f"  {output_base.with_suffix('.json')}")
    print(f"  {output_base.with_suffix('.srt')}")

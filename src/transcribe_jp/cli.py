from __future__ import annotations

import argparse
from pathlib import Path

from transcribe_jp.backends import transcribe_audio
from transcribe_jp.config import DEFAULT_LANGUAGE, DEFAULT_MODEL, TranscriptionConfig
from transcribe_jp.media import build_audio_path, build_ffmpeg_command, build_output_base, extract_audio
from transcribe_jp.transcript import write_transcript


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
    written = write_transcript(transcript, output_base)

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
    print(f"audio: {audio_path}")
    print("ffmpeg:")
    print("  " + " ".join(build_ffmpeg_command(media_path, audio_path)))
    print("outputs:")
    print(f"  {output_base.with_suffix('.txt')}")
    print(f"  {output_base.with_suffix('.json')}")
    print(f"  {output_base.with_suffix('.srt')}")

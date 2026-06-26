from __future__ import annotations

from pathlib import Path
import subprocess


def build_audio_path(media_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{media_path.stem}.wav"


def build_output_base(media_path: Path, output_dir: Path) -> Path:
    return output_dir / media_path.stem


def build_ffmpeg_command(media_path: Path, audio_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(audio_path),
    ]


def extract_audio(media_path: Path, audio_path: Path) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_ffmpeg_command(media_path, audio_path)
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg was not found. Install ffmpeg before running transcription.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed while extracting audio from {media_path}") from exc

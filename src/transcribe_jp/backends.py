from __future__ import annotations

from pathlib import Path
import shlex
import subprocess

from transcribe_jp.config import TranscriptionConfig
from transcribe_jp.transcript import Transcript, transcript_from_backend_output


def transcribe_audio(audio_path: Path, config: TranscriptionConfig) -> Transcript:
    if config.backend == "qwen-cli":
        output = _run_qwen_cli(audio_path, config)
    elif config.backend == "transformers":
        output = _run_transformers(audio_path, config)
    else:
        raise ValueError(f"Unsupported backend: {config.backend}")

    return transcript_from_backend_output(output, language=config.language, model=config.model)


def _run_qwen_cli(audio_path: Path, config: TranscriptionConfig) -> str:
    command = _format_command_template(config.resolved_command_template(), audio_path, config)
    try:
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "qwen-asr was not found. Install the Qwen ASR CLI or pass "
            "--command-template for your installed runner."
        ) from exc
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(f"Qwen ASR command failed: {details}") from exc
    return completed.stdout


def _run_transformers(audio_path: Path, config: TranscriptionConfig) -> dict[str, object]:
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "The transformers backend requires the gpu extra: "
            'python -m pip install -e ".[gpu]"'
        ) from exc

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline(
        "automatic-speech-recognition",
        model=config.model,
        torch_dtype=dtype,
        device=device,
        trust_remote_code=True,
    )
    result = asr(str(audio_path), return_timestamps=True, generate_kwargs={"language": config.language})
    if isinstance(result, dict):
        return result
    return {"text": str(result)}


def _format_command_template(
    template: str,
    audio_path: Path,
    config: TranscriptionConfig,
) -> list[str]:
    values = {
        "audio": str(audio_path),
        "model": config.model,
        "language": config.language,
    }
    return shlex.split(template.format(**values))

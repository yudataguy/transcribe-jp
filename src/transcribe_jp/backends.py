from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
from typing import Any

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
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3-ASR needs the gpu extra installed (Python 3.10+): "
            'python -m pip install -e ".[gpu]"'
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this backend on an NVIDIA GPU machine.")

    model = Qwen3ASRModel.from_pretrained(
        config.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=8,
        max_new_tokens=1024,
    )
    results = model.transcribe(
        audio=str(audio_path),
        language=_qwen_language(config.language),
    )
    result = results[0] if isinstance(results, list) else results
    return _qwen_result_to_mapping(result)


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


def _qwen_language(language: str) -> str | None:
    languages = {
        "auto": None,
        "ja": "Japanese",
        "japanese": "Japanese",
        "en": "English",
        "english": "English",
        "zh": "Chinese",
        "chinese": "Chinese",
    }
    return languages.get(language.lower(), language)


def _qwen_result_to_mapping(result: Any) -> dict[str, object]:
    text = getattr(result, "text", "")
    language = getattr(result, "language", None)
    timestamps = getattr(result, "time_stamps", None) or []
    segments = []

    for item in timestamps:
        segment = _timestamp_to_segment(item)
        if segment is not None:
            segments.append(segment)

    output: dict[str, object] = {"text": text}
    if language:
        output["detected_language"] = language
    if segments:
        output["segments"] = segments
    return output


def _timestamp_to_segment(item: Any) -> dict[str, object] | None:
    text = getattr(item, "text", None)
    start = getattr(item, "start_time", None)
    end = getattr(item, "end_time", None)

    if isinstance(item, dict):
        text = item.get("text", text)
        start = item.get("start_time", item.get("start", start))
        end = item.get("end_time", item.get("end", end))

    if text is None or start is None or end is None:
        return None
    return {"text": str(text), "start": float(start), "end": float(end)}

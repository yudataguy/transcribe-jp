from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import Any, Iterator

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
        import numpy as np
        import soundfile as sf
        import torch
        from tqdm import tqdm
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
        max_inference_batch_size=config.max_batch_size,
        max_new_tokens=1024,
    )

    audio, sample_rate = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    windows = list(_plan_windows(audio, sample_rate, config.window_seconds, np))
    language = _qwen_language(config.language)

    texts: list[str] = []
    segments: list[dict[str, object]] = []
    detected_language: str | None = None

    with tempfile.TemporaryDirectory(prefix="transcribe_jp_") as tmpdir:
        for index, (start, end) in enumerate(
            tqdm(windows, desc="Transcribing", unit="win")
        ):
            clip_path = Path(tmpdir) / f"window_{index:05d}.wav"
            sf.write(str(clip_path), audio[start:end], sample_rate)

            results = model.transcribe(audio=str(clip_path), language=language)
            result = results[0] if isinstance(results, list) else results
            mapping = _qwen_result_to_mapping(result)

            offset = start / sample_rate
            text = str(mapping.get("text", "")).strip()
            if text:
                texts.append(text)
            detected_language = detected_language or mapping.get("detected_language")  # type: ignore[assignment]

            window_segments = mapping.get("segments")
            if window_segments:
                for seg in window_segments:  # type: ignore[union-attr]
                    segments.append(
                        {
                            "text": seg["text"],
                            "start": float(seg["start"]) + offset,
                            "end": float(seg["end"]) + offset,
                        }
                    )
            elif text:
                # No fine-grained timestamps from the model: emit one coarse
                # segment per window so the .srt still has usable cues.
                segments.append({"text": text, "start": offset, "end": end / sample_rate})

    output: dict[str, object] = {"text": "\n".join(texts)}
    if detected_language:
        output["detected_language"] = detected_language
    if segments:
        output["segments"] = segments
    return output


def _plan_windows(
    audio: Any,
    sample_rate: int,
    window_seconds: float,
    np: Any,
    search_seconds: float = 3.0,
) -> Iterator[tuple[int, int]]:
    """Yield (start, end) sample ranges ~window_seconds long, cut on silence.

    Each target boundary is snapped to the quietest 25 ms frame within
    +/-search_seconds, so windows break during pauses instead of mid-word.
    """
    total = len(audio)
    window = max(1, int(window_seconds * sample_rate))
    search = int(search_seconds * sample_rate)
    frame = max(1, int(0.025 * sample_rate))

    start = 0
    while start < total:
        ideal_end = start + window
        if ideal_end >= total:
            yield (start, total)
            return

        lo = max(start + frame, ideal_end - search)
        hi = min(total, ideal_end + search)
        region = audio[lo:hi]
        num_frames = max(1, (hi - lo) // frame)
        energies = [
            float(np.sqrt(np.mean(np.square(region[j * frame : (j + 1) * frame]))))
            for j in range(num_frames)
        ]
        cut = lo + int(np.argmin(energies)) * frame + frame // 2
        cut = min(max(cut, start + frame), total)
        yield (start, cut)
        start = cut


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

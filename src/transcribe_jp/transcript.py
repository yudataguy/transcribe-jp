from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str = "ja"
    model: str = "Qwen/Qwen3-ASR-1.7B"
    segments: list[Segment] = field(default_factory=list)
    raw: dict[str, Any] | None = None


def transcript_from_backend_output(
    output: str | dict[str, Any],
    *,
    language: str,
    model: str,
) -> Transcript:
    if isinstance(output, str):
        stripped = output.strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return Transcript(text=stripped, language=language, model=model)
        return transcript_from_backend_output(parsed, language=language, model=model)

    text = str(output.get("text") or output.get("transcript") or "").strip()
    raw_segments = output.get("segments") or output.get("chunks") or []
    segments: list[Segment] = []

    for item in raw_segments:
        segment = _segment_from_mapping(item)
        if segment is not None:
            segments.append(segment)

    if not text and segments:
        text = "".join(segment.text for segment in segments).strip()

    return Transcript(text=text, language=language, model=model, segments=segments, raw=output)


def format_srt(transcript: Transcript) -> str:
    blocks = []
    for index, segment in enumerate(transcript.segments, start=1):
        text = segment.text.strip()
        if not text:
            continue
        blocks.append(
            f"{index}\n{_format_timestamp(segment.start)} --> {_format_timestamp(segment.end)}\n{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_transcript(transcript: Transcript, output_base: Path) -> dict[str, Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "txt": output_base.with_suffix(".txt"),
        "json": output_base.with_suffix(".json"),
        "srt": output_base.with_suffix(".srt"),
    }

    paths["txt"].write_text(transcript.text.strip() + "\n", encoding="utf-8")
    paths["json"].write_text(
        json.dumps(asdict(transcript), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["srt"].write_text(format_srt(transcript), encoding="utf-8")
    return paths


def _segment_from_mapping(item: Any) -> Segment | None:
    if not isinstance(item, dict):
        return None

    start = item.get("start")
    end = item.get("end")
    if (start is None or end is None) and isinstance(item.get("timestamp"), list):
        timestamp = item["timestamp"]
        if len(timestamp) >= 2:
            start, end = timestamp[0], timestamp[1]

    text = item.get("text") or item.get("transcript") or ""
    try:
        return Segment(start=float(start), end=float(end), text=str(text).strip())
    except (TypeError, ValueError):
        return None


def _format_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

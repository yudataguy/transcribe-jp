from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


# Subtitle grouping defaults. Fine-grained aligner timestamps are word/phrase
# level; these merge them into readable, sentence-length .srt cues.
SRT_MAX_CHARS = 36
SRT_MAX_DURATION = 6.0
SRT_MAX_GAP = 1.0
SRT_LINE_WIDTH = 21
_SENTENCE_ENDERS = "。．！？!?…"


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
    for index, segment in enumerate(group_segments_for_srt(transcript.segments), start=1):
        text = _wrap_for_display(segment.text)
        if not text:
            continue
        blocks.append(
            f"{index}\n{_format_timestamp(segment.start)} --> {_format_timestamp(segment.end)}\n{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def group_segments_for_srt(
    segments: list[Segment],
    *,
    max_chars: int = SRT_MAX_CHARS,
    max_duration: float = SRT_MAX_DURATION,
    max_gap: float = SRT_MAX_GAP,
) -> list[Segment]:
    """Merge word/phrase fragments into sentence-length subtitle cues.

    Fragments are accumulated until a natural break: the running text ends a
    sentence, the pause before the next fragment exceeds ``max_gap``, or adding
    it would exceed ``max_chars`` or ``max_duration``. Cues are only merged,
    never split, so already-large segments pass through unchanged.
    """
    cues: list[Segment] = []
    start = 0.0
    end = 0.0
    text = ""

    for segment in segments:
        piece = segment.text.strip()
        if not piece:
            continue
        if not text:
            start, end, text = segment.start, segment.end, piece
            continue

        merged = _join_fragment(text, piece)
        ends_sentence = text[-1] in _SENTENCE_ENDERS
        if (
            ends_sentence
            or (segment.start - end) > max_gap
            or len(merged) > max_chars
            or (segment.end - start) > max_duration
        ):
            cues.append(Segment(start=start, end=end, text=text))
            start, end, text = segment.start, segment.end, piece
        else:
            text, end = merged, segment.end

    if text:
        cues.append(Segment(start=start, end=end, text=text))
    return cues


def _join_fragment(left: str, right: str) -> str:
    # CJK text has no inter-word spaces; only add one between ASCII words so
    # mixed-in latin (names, acronyms) stays readable.
    if left[-1:].isascii() and left[-1:].isalnum() and right[:1].isascii() and right[:1].isalnum():
        return f"{left} {right}"
    return left + right


def _wrap_for_display(text: str, width: int = SRT_LINE_WIDTH) -> str:
    text = text.strip()
    if len(text) <= width:
        return text
    if " " in text:
        lines: list[str] = []
        current = ""
        for word in text.split(" "):
            if current and len(current) + 1 + len(word) > width:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}" if current else word
        if current:
            lines.append(current)
    else:
        lines = [text[i : i + width] for i in range(0, len(text), width)]
    return "\n".join(lines)


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

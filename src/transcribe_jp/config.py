from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_LANGUAGE = "ja"
DEFAULT_MAX_BATCH_SIZE = 2
DEFAULT_WINDOW_SECONDS = 60.0
DEFAULT_FORCED_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"
DEFAULT_ATTN_IMPL = "sdpa"


@dataclass(frozen=True)
class TranscriptionConfig:
    model: str = DEFAULT_MODEL
    language: str = DEFAULT_LANGUAGE
    output_dir: Path = Path("outputs")
    backend: str = "transformers"
    keep_audio: bool = False
    command_template: str | None = None
    # Number of audio chunks the transformers backend decodes in parallel.
    # Attention memory scales with this; lower it to 1 if you still hit CUDA OOM.
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE
    # Target length of each silence-aligned audio window, in seconds. Drives the
    # progress bar granularity and also bounds per-window attention memory.
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    # Optional forced-aligner model id for word/phrase-level .srt timestamps.
    # None loads only the ASR model; coarse per-window cues are used instead.
    forced_aligner: str | None = None
    # Attention kernel: "sdpa" (default), "flash_attention_2" (needs flash-attn,
    # faster + lighter on Ampere+), or "eager".
    attn_implementation: str = DEFAULT_ATTN_IMPL

    def resolved_command_template(self) -> str:
        if self.command_template:
            return self.command_template
        return "qwen-asr --model {model} --language {language} --audio {audio}"

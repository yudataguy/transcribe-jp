from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_LANGUAGE = "ja"
DEFAULT_MAX_BATCH_SIZE = 2
DEFAULT_WINDOW_SECONDS = 60.0


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

    def resolved_command_template(self) -> str:
        if self.command_template:
            return self.command_template
        return "qwen-asr --model {model} --language {language} --audio {audio}"

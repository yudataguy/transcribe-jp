from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_LANGUAGE = "ja"


@dataclass(frozen=True)
class TranscriptionConfig:
    model: str = DEFAULT_MODEL
    language: str = DEFAULT_LANGUAGE
    output_dir: Path = Path("outputs")
    backend: str = "transformers"
    keep_audio: bool = False
    command_template: str | None = None

    def resolved_command_template(self) -> str:
        if self.command_template:
            return self.command_template
        return "qwen-asr --model {model} --language {language} --audio {audio}"

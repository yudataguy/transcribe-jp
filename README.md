# transcribe-jp

Portable Japanese video transcription setup for `Qwen/Qwen3-ASR-1.7B`.

This repo is designed so the project can be prepared on a laptop that cannot run the model, then copied to a GPU machine for inference. Local tests and `--dry-run` do not download model weights.

## What It Does

- Extracts mono 16 kHz WAV audio from a video with `ffmpeg`.
- Runs Japanese ASR with `Qwen/Qwen3-ASR-1.7B`.
- Writes `.txt`, `.json`, and `.srt` files under `outputs/`.
- Supports two backends:
  - `transformers` for Qwen's `qwen-asr` package using its transformers backend.
  - `qwen-cli` for an external Qwen ASR command, with a configurable command template.

## GPU Machine Setup

Use Python 3.10 or newer on the machine that will run inference. The `qwen-asr` package supports Python 3.9–3.13, so Lambda Cloud's default Python 3.10 works without installing a newer interpreter.

```bash
cd transcribe-jp
./scripts/bootstrap_gpu.sh
```

Install `ffmpeg` if it is not already present:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg
```

## Run Transcription

```bash
source .venv/bin/activate
transcribe-jp /path/to/video.mp4 --output-dir outputs --backend transformers
```

Expected outputs:

```text
outputs/video.txt
outputs/video.json
outputs/video.srt
```

If your installed Qwen ASR package exposes a CLI, use the `qwen-cli` backend:

```bash
transcribe-jp /path/to/video.mp4 \
  --backend qwen-cli \
  --command-template 'qwen-asr --model {model} --language {language} --audio {audio}'
```

The placeholders available in `--command-template` are `{model}`, `{language}`, and `{audio}`.

## Dry Run On Any Machine

This checks paths and planned commands without requiring CUDA, model weights, or valid media content.

```bash
PYTHONPATH=src python3 -m transcribe_jp /path/to/video.mp4 --dry-run
```

After installing the package into a venv:

```bash
transcribe-jp /path/to/video.mp4 --dry-run
```

## Local Verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall src tests
```

## Notes

- The default model is `Qwen/Qwen3-ASR-1.7B`.
- The default language is `ja`.
- The `transformers` backend intentionally uses Qwen's `qwen-asr` Python package. Do not replace it with generic `transformers.pipeline`; generic Transformers releases may not recognize the `qwen3_asr` architecture yet.
- SRT timestamps are written when the backend returns timestamped chunks or segments. If the backend returns plain text only, `.txt` and `.json` will still contain the transcript and `.srt` will be empty.
- The first real run on the GPU machine will download model weights through Hugging Face unless they are already cached.

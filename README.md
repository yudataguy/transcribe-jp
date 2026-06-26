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

## Reduce Upload Time

The model only consumes 16 kHz mono audio, so there is no need to upload the
full video to the GPU machine. Extract compact audio locally first:

```bash
./scripts/extract_audio.sh videos/SDDE-652.mp4        # -> SDDE-652.opus (~100-300x smaller)
./scripts/extract_audio.sh -f mp3 videos/SDDE-652.mp4 # -> SDDE-652.mp3
```

Defaults to Opus 16 kbps; override with `-f opus|mp3|wav` and `-b BITRATE`. Upload
the resulting audio file and pass it to `transcribe-jp` exactly like a video.

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
- The `transformers` backend splits long audio into silence-aligned windows (`--window-seconds`, default 60) and shows a progress bar. Each window yields a coarse `.srt` cue by default.
- For word/phrase-level `.srt` timestamps, pass `--forced-aligner` (loads a second ~0.6B model, e.g. `Qwen/Qwen3-ForcedAligner-0.6B`). Without it, cues are one-per-window.
- Aligner fragments are merged into sentence-length cues. Tune the grouping with `--srt-max-chars` (default 36), `--srt-max-duration` (6s), `--srt-max-gap` (1s), and `--srt-line-width` (21).
- If you still hit CUDA out-of-memory, lower `--max-batch-size` to 1 and/or `--window-seconds`, and set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- The first real run on the GPU machine will download model weights through Hugging Face unless they are already cached.

## Speed Tuning

Throughput is mostly bounded by how much of the GPU you use. Levers, by impact:

- **Raise `--max-batch-size`.** It was lowered to 1 only to fit a 22 GB A10. More VRAM means more parallel chunks:
  - A100 40/80 GB: try `--max-batch-size 16` (or 32 on 80 GB) with `--window-seconds 180`.
  - A10 22 GB: `--max-batch-size 4` without the aligner, `2` with it.
- **Use FlashAttention-2:** `pip install flash-attn --no-build-isolation`, then add `--attn-impl flash_attention_2`. Faster and lighter than the default `sdpa`, which lets you push batch size higher.
- **Drop `--forced-aligner` when you don't need word-level timing** — it runs a second model pass per window (~2x cost).
- **Bigger `--window-seconds`** means fewer per-call overheads; combine with FlashAttention-2 so the longer sequences still fit.
- TF32 matmul is enabled automatically on the transformers backend (free on Ampere+).
- For maximum throughput on an A100, the `qwen-asr` package also ships a vLLM path (`pip install "qwen-asr[vllm]"`); wiring it up is a larger change but offers the biggest gains.

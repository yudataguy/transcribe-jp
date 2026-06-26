# Command Template Examples

The `qwen-cli` backend delegates inference to an external command and reads transcript output from stdout.

Default:

```bash
transcribe-jp video.mp4 \
  --backend qwen-cli \
  --command-template 'qwen-asr --model {model} --language {language} --audio {audio}'
```

Wrapper script:

```bash
transcribe-jp video.mp4 \
  --backend qwen-cli \
  --command-template './run_qwen_asr.sh {model} {language} {audio}'
```

Your command should print either plain transcript text or JSON like:

```json
{
  "text": "こんにちは。",
  "segments": [
    {"start": 0.0, "end": 1.0, "text": "こんにちは。"}
  ]
}
```

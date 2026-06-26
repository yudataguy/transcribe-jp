import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


class CliTests(unittest.TestCase):
    def test_dry_run_prints_planned_steps_without_requiring_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "sample.mp4"
            media.write_bytes(b"not really a video")
            output_dir = Path(tmp) / "out"

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "transcribe_jp",
                    str(media),
                    "--output-dir",
                    str(output_dir),
                    "--dry-run",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        self.assertIn("Qwen/Qwen3-ASR-1.7B", completed.stdout)
        self.assertIn("language: ja", completed.stdout)
        self.assertIn("ffmpeg", completed.stdout)
        self.assertIn("sample.wav", completed.stdout)


if __name__ == "__main__":
    unittest.main()

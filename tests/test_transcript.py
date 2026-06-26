from pathlib import Path
import tempfile
import unittest

from transcribe_jp.transcript import Segment, Transcript, format_srt, write_transcript


class TranscriptTests(unittest.TestCase):
    def test_format_srt_writes_numbered_timestamped_segments(self) -> None:
        transcript = Transcript(
            text="こんにちは。今日はテストです。",
            segments=[
                Segment(start=0.0, end=1.5, text="こんにちは。"),
                Segment(start=61.25, end=63.0, text="今日はテストです。"),
            ],
        )

        self.assertEqual(
            format_srt(transcript),
            "1\n"
            "00:00:00,000 --> 00:00:01,500\n"
            "こんにちは。\n\n"
            "2\n"
            "00:01:01,250 --> 00:01:03,000\n"
            "今日はテストです。\n",
        )

    def test_write_transcript_creates_txt_json_and_srt(self) -> None:
        transcript = Transcript(
            text="こんにちは。",
            language="ja",
            model="Qwen/Qwen3-ASR-1.7B",
            segments=[Segment(start=0.0, end=1.0, text="こんにちは。")],
        )

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_transcript(transcript, Path(tmp) / "movie")

            self.assertEqual(set(paths), {"txt", "json", "srt"})
            self.assertEqual(paths["txt"].read_text(encoding="utf-8"), "こんにちは。\n")
            self.assertIn('"language": "ja"', paths["json"].read_text(encoding="utf-8"))
            self.assertIn("00:00:00,000 --> 00:00:01,000", paths["srt"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

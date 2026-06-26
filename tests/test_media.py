from pathlib import Path
import unittest

from transcribe_jp.media import build_audio_path, build_ffmpeg_command


class MediaTests(unittest.TestCase):
    def test_build_audio_path_uses_media_stem_and_wav_extension(self) -> None:
        output = build_audio_path(Path("videos/lecture.mp4"), Path("outputs"))

        self.assertEqual(output, Path("outputs/lecture.wav"))

    def test_build_ffmpeg_command_extracts_mono_16khz_wav(self) -> None:
        command = build_ffmpeg_command(Path("input.mp4"), Path("audio.wav"))

        self.assertEqual(
            command,
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                "input.mp4",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "audio.wav",
            ],
        )


if __name__ == "__main__":
    unittest.main()

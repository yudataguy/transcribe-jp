from types import SimpleNamespace
import unittest

from transcribe_jp.backends import _qwen_language, _qwen_result_to_mapping


class BackendTests(unittest.TestCase):
    def test_qwen_language_maps_ja_to_japanese(self) -> None:
        self.assertEqual(_qwen_language("ja"), "Japanese")
        self.assertIsNone(_qwen_language("auto"))

    def test_qwen_result_to_mapping_converts_timestamps_to_segments(self) -> None:
        result = SimpleNamespace(
            text="こんにちは。",
            language="Japanese",
            time_stamps=[
                SimpleNamespace(text="こんにちは。", start_time=0.0, end_time=1.25),
            ],
        )

        self.assertEqual(
            _qwen_result_to_mapping(result),
            {
                "text": "こんにちは。",
                "detected_language": "Japanese",
                "segments": [{"text": "こんにちは。", "start": 0.0, "end": 1.25}],
            },
        )


if __name__ == "__main__":
    unittest.main()

import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from transcribe_jp import backends
from transcribe_jp.backends import _qwen_language, _qwen_result_to_mapping
from transcribe_jp.config import TranscriptionConfig


class _FakeTqdm:
    """Minimal stand-in for tqdm used as a context manager with .update()."""

    def __init__(self, *args, **kwargs) -> None:
        self.updates: list[int] = []

    def __enter__(self) -> "_FakeTqdm":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def update(self, n: int) -> None:
        self.updates.append(n)


def _fake_result() -> SimpleNamespace:
    # One 0.5s segment starting at the window's local t=0, so the applied
    # offset equals the window's start time and is easy to assert.
    return SimpleNamespace(
        text="ok",
        language="Japanese",
        time_stamps=[SimpleNamespace(text="seg", start_time=0.0, end_time=0.5)],
    )


class _FakeModel:
    instance: "_FakeModel | None" = None

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, model, **kwargs):  # noqa: ANN001
        cls.instance = cls()
        return cls.instance

    def transcribe(self, audio, language=None, return_time_stamps=False):  # noqa: ANN001
        self.calls.append(
            {
                "batch_size": len(audio) if isinstance(audio, list) else 1,
                "language": language,
                "return_time_stamps": return_time_stamps,
            }
        )
        count = len(audio) if isinstance(audio, list) else 1
        return [_fake_result() for _ in range(count)]


class BatchedTransformersTests(unittest.TestCase):
    def setUp(self) -> None:
        torch = SimpleNamespace(
            bfloat16=object(),
            cuda=SimpleNamespace(is_available=lambda: True),
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                cudnn=SimpleNamespace(allow_tf32=False),
            ),
        )
        soundfile = SimpleNamespace(
            read=lambda path, dtype="float32": (np.zeros(80000, dtype="float32"), 16000),
            write=lambda *args, **kwargs: None,
        )
        tqdm_mod = SimpleNamespace(tqdm=_FakeTqdm)
        qwen_asr = SimpleNamespace(Qwen3ASRModel=_FakeModel)

        self._modules = mock.patch.dict(
            sys.modules,
            {
                "torch": torch,
                "soundfile": soundfile,
                "tqdm": tqdm_mod,
                "qwen_asr": qwen_asr,
            },
        )
        self._modules.start()
        _FakeModel.instance = None

        # Fixed windows so offsets are deterministic (sample_rate = 16000):
        # 0.0s, 1.0s, 3.0s.
        self._windows = mock.patch.object(
            backends,
            "_plan_windows",
            return_value=iter([(0, 16000), (16000, 48000), (48000, 80000)]),
        )
        self._windows.start()

    def tearDown(self) -> None:
        self._windows.stop()
        self._modules.stop()

    def test_windows_are_decoded_in_batches(self) -> None:
        config = TranscriptionConfig(max_batch_size=2, forced_aligner="X")
        backends._run_transformers(mock.Mock(), config)

        calls = _FakeModel.instance.calls
        # 3 windows at batch size 2 -> a full batch of 2 then a batch of 1.
        self.assertEqual([c["batch_size"] for c in calls], [2, 1])

    def test_forced_aligner_requests_timestamps(self) -> None:
        config = TranscriptionConfig(max_batch_size=8, forced_aligner="X")
        backends._run_transformers(mock.Mock(), config)

        self.assertTrue(all(c["return_time_stamps"] for c in _FakeModel.instance.calls))
        # 3 windows fit in one batch of 8.
        self.assertEqual([c["batch_size"] for c in _FakeModel.instance.calls], [3])

    def test_segment_offsets_track_each_windows_start(self) -> None:
        config = TranscriptionConfig(max_batch_size=2, forced_aligner="X")
        output = backends._run_transformers(mock.Mock(), config)

        starts = [round(seg["start"], 3) for seg in output["segments"]]
        ends = [round(seg["end"], 3) for seg in output["segments"]]
        # Each window's local 0.0-0.5s segment shifted by its start offset.
        self.assertEqual(starts, [0.0, 1.0, 3.0])
        self.assertEqual(ends, [0.5, 1.5, 3.5])


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

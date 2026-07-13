import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from round7.alignment import (
    AlignmentError,
    cached_alignment_result,
    forced_align_ctc,
    group_words_into_segments,
    speech_occupancy,
    token_alignment_to_words,
)


class ForcedAlignmentTests(unittest.TestCase):
    def test_repeated_tokens_require_blank_state(self):
        # Best path: blank, a, blank, a, blank, b, blank.
        path = [0, 1, 0, 1, 0, 2, 0]
        logits = torch.full((len(path), 4), -8.0)
        for frame, token in enumerate(path):
            logits[frame, token] = 8.0
        alignment = forced_align_ctc(logits.log_softmax(dim=-1), [1, 1, 2], blank_id=0)
        self.assertEqual([item["start_frame"] for item in alignment], [1, 3, 5])
        self.assertTrue(all(item["mean_posterior"] > 0.99 for item in alignment))

    def test_too_few_frames_is_rejected(self):
        with self.assertRaisesRegex(AlignmentError, "frames"):
            forced_align_ctc(torch.zeros(2, 4), [1, 2, 3], blank_id=0)

    def test_word_timestamps_follow_delimiters(self):
        targets = [1, 2, 3, 4]
        aligned = [
            {"start_frame": index, "end_frame": index + 1, "mean_posterior": 0.8}
            for index in range(4)
        ]
        words = token_alignment_to_words(
            "ab c", targets, aligned, np.array([0.1, 0.2, 0.3, 0.4]), delimiter_id=3
        )
        self.assertEqual([word["word"] for word in words], ["ab", "c"])
        self.assertAlmostEqual(words[1]["start"], 0.4)

    def test_segment_grouping_obeys_maximum_duration(self):
        words = [
            {"word": "one", "start": 0.2, "end": 1.0, "mean_token_posterior": 0.9},
            {"word": "two", "start": 1.1, "end": 2.0, "mean_token_posterior": 0.8},
            {"word": "three", "start": 5.0, "end": 6.0, "mean_token_posterior": 0.7},
        ]
        segments = group_words_into_segments(words, 7.0, 0.5, 3.0, 0.1)
        self.assertEqual([segment["text"] for segment in segments], ["one two", "three"])
        self.assertTrue(all(segment["end"] - segment["start"] <= 3.0 for segment in segments))

    def test_speech_occupancy_distinguishes_silence(self):
        self.assertEqual(speech_occupancy(np.zeros(16000, dtype=np.float32), 16000), 0.0)
        tone = np.sin(2 * np.pi * 220 * np.arange(16000) / 16000).astype(np.float32)
        self.assertGreater(speech_occupancy(tone, 16000), 0.9)

    def test_cached_result_requires_accepted_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "segment.flac"
            result = root / "result.json"
            result.write_text(
                json.dumps({"accepted": [{"audio_path": str(audio)}], "rejected": []}),
                encoding="utf-8",
            )
            self.assertIsNone(cached_alignment_result(result))
            audio.touch()
            self.assertIsNotNone(cached_alignment_result(result))


if __name__ == "__main__":
    unittest.main()

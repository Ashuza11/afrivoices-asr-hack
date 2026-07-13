import json
import tempfile
import unittest
from pathlib import Path

from round7.data import LANGUAGES
from round7.training import (
    _latest_checkpoint,
    _learning_rate_lambda,
    _prune_checkpoints,
    language_balanced_sampler,
)


class TrainingControlTests(unittest.TestCase):
    def test_learning_rate_warms_up_and_decays(self):
        self.assertAlmostEqual(_learning_rate_lambda(50, 100, 1000), 0.5)
        self.assertAlmostEqual(_learning_rate_lambda(100, 100, 1000), 1.0)
        self.assertAlmostEqual(_learning_rate_lambda(1000, 100, 1000), 0.0)

    def test_balanced_sampler_requires_all_languages(self):
        records = [
            {"language": language, "duration": 10.0}
            for language in LANGUAGES
        ]
        self.assertEqual(len(list(language_balanced_sampler(records, 7))), len(LANGUAGES))
        with self.assertRaisesRegex(RuntimeError, "no records"):
            language_balanced_sampler(records[:-1], 7)

    def test_latest_pointer_and_pruning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "checkpoint-0000001"
            second = root / "checkpoint-0000002"
            first.mkdir()
            second.mkdir()
            (first / "state.json").write_text("{}", encoding="utf-8")
            (second / "state.json").write_text("{}", encoding="utf-8")
            (root / "latest.json").write_text(
                json.dumps({"checkpoint": str(second.resolve())}), encoding="utf-8"
            )
            self.assertEqual(_latest_checkpoint(root), second.resolve())
            _prune_checkpoints(root, {second})
            self.assertFalse(first.exists())
            self.assertTrue(second.exists())


if __name__ == "__main__":
    unittest.main()

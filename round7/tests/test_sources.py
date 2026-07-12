import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from round7.sources import (
    ANV_LANGUAGES,
    clean_optional,
    decode_audio_field,
    list_anv_shards,
    manifest_audio_complete,
    stable_id,
    validate_source_policy,
)


class SourcePolicyTests(unittest.TestCase):
    def valid_config(self):
        return {
            "sources": {
                "policy": {
                    "digital_umuganda_languages": ["swa"],
                    "anv_languages": list(ANV_LANGUAGES),
                    "allow_digital_umuganda_somali": False,
                },
                "anv": {"languages": list(ANV_LANGUAGES)},
            }
        }

    def test_policy_locks_somali_to_anv(self):
        validate_source_policy(self.valid_config())
        invalid = self.valid_config()
        invalid["sources"]["policy"]["allow_digital_umuganda_somali"] = True
        with self.assertRaisesRegex(ValueError, "Somali must remain disabled"):
            validate_source_policy(invalid)

    def test_stable_id_is_repeatable(self):
        self.assertEqual(stable_id("som", "a", 3), stable_id("som", "a", 3))
        self.assertNotEqual(stable_id("som", "a", 3), stable_id("som", "a", 4))

    def test_array_audio_is_resampled(self):
        audio = decode_audio_field(
            {"array": np.zeros(8000, dtype=np.float32), "sampling_rate": 8000}, 16000
        )
        self.assertEqual(len(audio), 16000)

    def test_pandas_nan_is_not_treated_as_metadata(self):
        self.assertIsNone(clean_optional(float("nan")))

    def test_cached_manifest_requires_its_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            audio = root / "clip.flac"
            manifest.write_text(
                json.dumps({"audio_path": str(audio)}) + "\n", encoding="utf-8"
            )
            self.assertFalse(manifest_audio_complete(manifest))
            audio.touch()
            self.assertTrue(manifest_audio_complete(manifest))

    @patch("round7.sources.list_repo_files")
    def test_anv_limit_is_spread_across_bucket(self, mocked_list):
        mocked_list.return_value = [
            f"som/train/unscripted/audios/shard_{index:02d}.parquet" for index in range(10)
        ]
        config = {
            "repo_id": "test/repo",
            "languages": ["som"],
            "split": "train",
            "subtypes": ["unscripted"],
            "maximum_shards_per_bucket": 3,
        }
        selected = list_anv_shards(config, token=None)
        self.assertEqual([Path(path).stem for _, _, path in selected], ["shard_00", "shard_04", "shard_09"])


if __name__ == "__main__":
    unittest.main()

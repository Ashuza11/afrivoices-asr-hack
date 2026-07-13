import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from round7.data import (
    LANGUAGES,
    ManifestError,
    assert_no_group_leakage,
    assign_grouped_splits,
    build_shared_vocabulary,
    canonicalize_record,
    load_and_validate_manifest,
    normalize_text,
    verify_audio_records,
)
from round7.pipeline import load_yaml, unresolved_fields


class NormalizationTests(unittest.TestCase):
    def test_round6_winning_normalization(self):
        self.assertEqual(normalize_text("  HÉLLO,  Gĩkũyũ! "), "héllo gĩkũyũ")


class ManifestTests(unittest.TestCase):
    def test_aliases_are_canonicalized(self):
        record = canonicalize_record(
            {
                "record_id": "r1",
                "recording_id": "source-1",
                "path": "/audio/a.flac",
                "lang": "KIK",
                "actualSentence": "Nĩ wega!",
                "seconds": "4.5",
            },
            1,
        )
        self.assertEqual(record["language"], "kik")
        self.assertEqual(record["transcription"], "nĩ wega")
        self.assertEqual(record["duration"], 4.5)

    def test_rejections_are_reported_without_hiding_valid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            rows = [
                {
                    "id": "good",
                    "source_id": "source-good",
                    "audio_path": "/a.flac",
                    "language": "swa",
                    "transcription": "habari",
                },
                {
                    "id": "bad",
                    "source_id": "source-bad",
                    "audio_path": "/b.flac",
                    "language": "xxx",
                    "transcription": "bad",
                },
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            accepted, rejected = load_and_validate_manifest(path)
            self.assertEqual(len(accepted), 1)
            self.assertEqual(len(rejected), 1)
            self.assertIn("unsupported language", rejected[0]["reason"])

    def test_shared_vocabulary_uses_training_text_only(self):
        vocabulary = build_shared_vocabulary(
            [
                {"split": "train", "transcription": "ab c"},
                {"split": "validation", "transcription": "z"},
            ]
        )
        self.assertEqual(vocabulary["<pad>"], 0)
        self.assertEqual(vocabulary["|"], 2)
        self.assertNotIn("z", vocabulary)

    def test_audio_verification_reads_metadata_without_loading_pipeline_state(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "sample.wav"
            sf.write(audio_path, np.zeros(16000, dtype=np.float32), 16000)
            accepted, rejected = verify_audio_records(
                [
                    {
                        "id": "sample",
                        "audio_path": "sample.wav",
                        "duration": None,
                    }
                ],
                Path(directory),
                minimum_duration=0.5,
                maximum_duration=2.0,
            )
            self.assertFalse(rejected)
            self.assertAlmostEqual(accepted[0]["duration"], 1.0)
            self.assertEqual(accepted[0]["sample_rate"], 16000)


class SplitTests(unittest.TestCase):
    def make_records(self):
        records = []
        for language in LANGUAGES:
            for source_index in range(4):
                for segment_index in range(2):
                    records.append(
                        {
                            "id": f"{language}-{source_index}-{segment_index}",
                            "source_id": f"{language}-source-{source_index}",
                            "speaker_id": f"{language}-speaker-{source_index}",
                            "language": language,
                            "subtype": "unscripted",
                            "duration": 10.0,
                            "split": None,
                        }
                    )
        return records

    def test_split_is_deterministic_and_source_safe(self):
        first = assign_grouped_splits(self.make_records(), 0.25, 42)
        second = assign_grouped_splits(self.make_records(), 0.25, 42)
        self.assertEqual(first, second)
        assert_no_group_leakage(first)
        for language in LANGUAGES:
            splits = {row["split"] for row in first if row["language"] == language}
            self.assertEqual(splits, {"train", "validation"})

    def test_split_rejects_language_with_one_source(self):
        records = self.make_records()
        records = [row for row in records if row["language"] != "mas" or row["source_id"].endswith("0")]
        with self.assertRaises(ManifestError):
            assign_grouped_splits(records, 0.25, 42)

    def test_same_speaker_cannot_cross_splits(self):
        records = self.make_records()
        for row in records:
            if row["language"] == "kik" and row["source_id"] in {
                "kik-source-0",
                "kik-source-1",
            }:
                row["speaker_id"] = "shared-kik-speaker"
        split = assign_grouped_splits(records, 0.25, 42)
        speaker_splits = {
            row["split"] for row in split if row.get("speaker_id") == "shared-kik-speaker"
        }
        self.assertEqual(len(speaker_splits), 1)


class ClusterConfigTests(unittest.TestCase):
    def test_unresolved_fields_are_named(self):
        self.assertEqual(
            unresolved_fields({"scheduler": {"partition": "FILL_ME", "qos": None}}),
            ["scheduler.partition"],
        )

    def test_hex_profile_inherits_safe_a5000_settings(self):
        config = load_yaml(Path("round7/config.hex.yaml"))
        self.assertEqual(config["training"]["per_device_batch_size"], 1)
        self.assertEqual(config["training"]["gradient_accumulation_steps"], 8)
        self.assertEqual(config["paths"]["scratch_dir"], "outputs/round7/work")
        cluster = load_yaml(Path("round7/cluster.hex.yaml"))
        self.assertFalse(unresolved_fields(cluster))
        self.assertEqual(cluster["scheduler"]["partition"], "a5000-48h")
        self.assertEqual(cluster["resources"]["gpu_count"], 1)


if __name__ == "__main__":
    unittest.main()

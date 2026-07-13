import unittest

from round7.data import LANGUAGES
from round7.gate import (
    alignment_hours,
    assert_no_alignment_leakage,
    evaluate_gate,
    select_lower_macro_candidate,
)


class AlignmentGateTests(unittest.TestCase):
    def metrics(self, macro, values):
        return {"macro_wer": macro, **{f"wer_{lang}": value for lang, value in zip(LANGUAGES, values)}}

    def test_gate_accepts_material_broad_improvement(self):
        clean = self.metrics(0.70, [0.7] * 6)
        aligned = self.metrics(0.65, [0.65, 0.66, 0.64, 0.68, 0.63, 0.64])
        config = {
            "required_alignment_languages": ["som", "kik", "luo", "mas", "kln"],
            "minimum_accepted_hours_per_language": 1,
            "minimum_macro_wer_improvement": 0.02,
            "minimum_languages_improved": 4,
            "maximum_language_regression": 0.03,
        }
        report = evaluate_gate(clean, aligned, {lang: 2 for lang in LANGUAGES}, config)
        self.assertTrue(report["proceed"])
        self.assertEqual(report["languages_improved"], 6)

    def test_gate_rejects_regression_and_missing_hours(self):
        clean = self.metrics(0.70, [0.7] * 6)
        aligned = self.metrics(0.69, [0.75, 0.68, 0.68, 0.68, 0.68, 0.67])
        config = {
            "required_alignment_languages": ["som", "kik", "luo", "mas", "kln"],
            "minimum_accepted_hours_per_language": 1,
            "minimum_macro_wer_improvement": 0.02,
            "minimum_languages_improved": 4,
            "maximum_language_regression": 0.03,
        }
        hours = {lang: 2 for lang in LANGUAGES}
        hours["som"] = 0.5
        report = evaluate_gate(clean, aligned, hours, config)
        self.assertFalse(report["proceed"])
        self.assertTrue(any(reason.startswith("insufficient") for reason in report["reasons"]))
        self.assertTrue(any(reason.startswith("excessive") for reason in report["reasons"]))

    def test_alignment_hours_and_leakage(self):
        rows = [{"language": lang, "duration": 3600, "source_id": lang} for lang in LANGUAGES]
        self.assertEqual(alignment_hours(rows), {lang: 1 for lang in LANGUAGES})
        with self.assertRaisesRegex(RuntimeError, "leaks"):
            assert_no_alignment_leakage(
                [{"language": "swa", "source_id": "swa", "split": "validation"}], rows
            )

    def test_lower_macro_candidate_is_selected_for_inference(self):
        pilot = {"name": "pilot", "checkpoint": "a", "metrics": {"macro_wer": 0.51}}
        final = {"name": "final", "checkpoint": "b", "metrics": {"macro_wer": 0.55}}
        self.assertEqual(select_lower_macro_candidate(pilot, final)["checkpoint"], "a")


if __name__ == "__main__":
    unittest.main()

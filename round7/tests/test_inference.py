import unittest

import pandas as pd

from round7.inference import validate_submission


class SubmissionTests(unittest.TestCase):
    def test_complete_submission_passes(self):
        expected = pd.DataFrame({"id": ["1", "2"], "language": ["swa", "som"]})
        submission = expected.assign(transcription=["habari", "nabad"])
        validate_submission(submission, expected)

    def test_missing_or_empty_predictions_fail(self):
        expected = pd.DataFrame({"id": ["1", "2"], "language": ["swa", "som"]})
        missing = pd.DataFrame({"id": ["1"], "language": ["swa"], "transcription": ["x"]})
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            validate_submission(missing, expected)
        empty = expected.assign(transcription=["x", ""])
        with self.assertRaisesRegex(RuntimeError, "empty"):
            validate_submission(empty, expected)


if __name__ == "__main__":
    unittest.main()

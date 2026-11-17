import copy
import tempfile
import unittest
from pathlib import Path

from snufspin.evaluation import evaluate_review
from snufspin.tracking import BallPoint, track_report
from snufspin.video import Video


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "source.mp4"
        path.write_bytes(b"test video")
        self.video = Video(path, 320, 180, (0.0, 0.1, 0.2, 0.3, 0.4))
        self.report = track_report(self.video, (
            BallPoint(0, 13, 14, "measured"),
            BallPoint(2, 20, 20, "predicted"),
            BallPoint(4, 80, 80, "measured"),
        ), {"engine": "test"})
        self.review = {
            "video_sha256": self.report["video_sha256"],
            "corrections": [
                {"frame": 0, "timestamp_s": 0, "status": "visible", "x_px": 10, "y_px": 10},
                {"frame": 1, "timestamp_s": 0.1, "status": "visible", "x_px": 20, "y_px": 20},
                {"frame": 2, "timestamp_s": 0.2, "status": "hidden"},
                {"frame": 3, "timestamp_s": 0.3, "status": "hidden"},
            ],
        }

    def test_counts_only_reviewed_frames_and_preserves_provenance(self):
        result = evaluate_review(self.video, self.report, self.review)
        self.assertEqual(result["reviewed_frames"], 4)
        self.assertEqual(result["visible_frames"], 2)
        self.assertEqual(result["visible_without_position"], 1)
        self.assertEqual(result["hidden_with_position"], 1)
        self.assertEqual(result["localized_within_px"], {"5": 1, "10": 1, "20": 1})
        self.assertEqual(result["mean_error_px_when_position_present"], 5)
        self.assertEqual(result["frames"][2]["provenance"], "predicted")

    def test_rejects_mismatched_or_invalid_review(self):
        changes = (
            {"frame": 5}, {"frame": True}, {"timestamp_s": 0.04},
            {"timestamp_s": float("nan")}, {"x_px": 320},
            {"x_px": float("nan")}, {"status": "unsure"},
        )
        for change in changes:
            with self.subTest(change=change):
                review = copy.deepcopy(self.review)
                review["corrections"][0].update(change)
                with self.assertRaises(ValueError):
                    evaluate_review(self.video, self.report, review)
        with self.assertRaisesRegex(ValueError, "different video"):
            evaluate_review(self.video, self.report, self.review | {"video_sha256": "wrong"})
        with self.assertRaisesRegex(ValueError, "at least one"):
            evaluate_review(self.video, self.report, self.review | {"corrections": []})
        duplicate = self.review["corrections"] * 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_review(self.video, self.report, self.review | {"corrections": duplicate})

    def test_all_hidden_has_no_localization_error(self):
        review = self.review | {"corrections": self.review["corrections"][2:]}
        result = evaluate_review(self.video, self.report, review)
        self.assertIsNone(result["mean_error_px_when_position_present"])

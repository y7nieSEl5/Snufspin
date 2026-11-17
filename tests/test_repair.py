import copy
import tempfile
import unittest
from pathlib import Path

from snufspin.repair import interpolate_short_gaps
from snufspin.tracking import BallPoint, read_ball_track, track_report
from snufspin.video import Video


class RepairTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        source = Path(directory.name) / "source.mp4"
        source.write_bytes(b"video")
        self.video = Video(source, 320, 180, (0, 0.01, 0.03, 0.1, 0.2, 0.3))

    def test_uses_elapsed_time_and_preserves_measured_points_and_edge_gaps(self):
        before = BallPoint(1, 10, 10, "measured", 0.9)
        after = BallPoint(3, 100, 100, "manual")
        report = track_report(self.video, (before, after), {"engine": "test"})
        unchanged = copy.deepcopy(report)
        result = interpolate_short_gaps(self.video, report)
        points = read_ball_track(self.video, result)
        self.assertEqual([p.frame for p in points], [1, 2, 3])
        self.assertEqual(points[0], before)
        self.assertEqual(points[2], after)
        self.assertAlmostEqual(points[1].x_px, 30)
        self.assertEqual(points[1].provenance, "interpolated")
        self.assertIsNone(points[1].confidence)
        self.assertEqual(report, unchanged)
        self.assertEqual(result, interpolate_short_gaps(self.video, result))

    def test_leaves_long_gaps_and_existing_inferred_samples_alone(self):
        points = (
            BallPoint(0, 10, 10, "measured"),
            BallPoint(2, 30, 30, "predicted"),
            BallPoint(3, 40, 40, "measured"),
            BallPoint(5, 60, 60, "measured"),
        )
        report = track_report(self.video, points, {"engine": "test"})
        self.assertEqual(read_ball_track(self.video, interpolate_short_gaps(self.video, report)), points)

    def test_rejects_invalid_gap_limit(self):
        report = track_report(self.video, (), {"engine": "test"})
        for limit in (0, -1, float("nan"), float("inf")):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                interpolate_short_gaps(self.video, report, limit)

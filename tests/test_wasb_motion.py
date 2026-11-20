from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from snufspin.tracking import BallPoint, track_report
from snufspin.video import Video
from snufspin.wasb_motion import MotionSettings, replay_wasb


def candidate(x, y=150, score=1.0, passed_gate=True):
    return {"x_px": x, "y_px": y, "score": score, "passed_gate": passed_gate}


class WasbMotionTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "source.mp4"
        self.path.write_bytes(b"synthetic video identity")
        times = tuple(frame / 60 + 0.002 * (frame % 3) for frame in range(25))
        self.video = Video(self.path, 1920, 1088, times)
        self.frame = 12

    def fixture(self, trajectory, candidates, winner, gaps=()):
        points = [
            BallPoint(frame, *trajectory(frame, time), "measured")
            for frame, time in enumerate(self.video.timestamps_s)
            if frame != self.frame and frame not in gaps
        ]
        if winner is not None:
            selected = candidates[winner]
            points.append(BallPoint(self.frame, selected["x_px"], selected["y_px"], "measured"))
        baseline = track_report(self.video, tuple(points), {
            "engine": "wasb-tennis", "input_width": 512, "heatmap_scale": 0,
        })
        log = {key: value for key, value in baseline.items() if key != "ball_track"}
        log["frames"] = [{
            "frame": self.frame, "timestamp_s": self.video.time_at(self.frame),
            "candidates": candidates, "selected_index": winner,
        }]
        return baseline, log

    def replay(self, fixture, mode, settings=MotionSettings()):
        report, rows = replay_wasb(self.video, *fixture, mode, settings)
        self.assertEqual(report["source"]["tolerance_source_px"], 11.25)
        return report, rows[0]

    def test_robust_fit_ignores_mask_and_side_outliers_and_uses_timestamps(self):
        time = self.video.time_at(self.frame)
        x = 100 + 210 * time

        def line_with_outliers(frame, t):
            return (600, 300) if frame in {4, 11, 13, 19} else (100 + 210 * t, 150)

        fixture = self.fixture(line_with_outliers, [
            candidate(x + 30, score=100),
            candidate(x + 0.05, score=2, passed_gate=False),
            candidate(x + 0.25, score=4, passed_gate=False),
        ], winner=0)
        _, rejected = self.replay(fixture, "reject")
        reranked, chosen = self.replay(fixture, "rerank")
        self.assertEqual(rejected["check"], "straight")
        self.assertIsNone(rejected["selected_index"])
        self.assertEqual(chosen["selected_index"], 2)
        self.assertAlmostEqual(chosen["candidate_errors_px"][0], 30)
        points = {point["frame"]: point for point in reranked["ball_track"]}
        self.assertEqual(points[self.frame]["x_px"], x + 0.25)
        self.assertEqual(points[self.frame]["provenance"], "measured")
        self.assertEqual(points[11]["x_px"], 600)

    def test_keep_winner_when_it_fits_both_sides(self):
        def line(frame, t):
            return (100 if frame < self.frame else 106, 150)

        fixture = self.fixture(line, [candidate(101), candidate(103, score=10)], winner=0)
        _, row = self.replay(fixture, "rerank")
        self.assertEqual(row["selected_index"], 0)
        self.assertAlmostEqual(row["candidate_errors_px"][0], 5)

    def test_bend_uses_incoming_before_contact_and_outgoing_after(self):
        self.video = Video(self.path, 1920, 1088, tuple(frame / 60 for frame in range(25)))
        contact_frame = 12

        def bounce(frame, t):
            return (200 - 8 * abs(frame - contact_frame), 150)

        for target in (10, 14):
            with self.subTest(target=target):
                self.frame = target
                expected = bounce(target, self.video.time_at(target))[0]
                fixture = self.fixture(bounce, [candidate(expected)], winner=0)
                _, row = self.replay(fixture, "reject")
                self.assertEqual(row["check"], "bend")
                self.assertAlmostEqual(row["contact_time_s"], self.video.time_at(contact_frame))
                self.assertEqual(row["selected_index"], 0)

    def test_parallel_disagreement_is_inconclusive(self):
        fixture = self.fixture(
            lambda frame, t: (100 + t * 30, 100 if frame < self.frame else 160),
            [candidate(500)], winner=0,
        )
        _, row = self.replay(fixture, "rerank")
        self.assertEqual(row["check"], "inconclusive")
        self.assertEqual(row["selected_index"], 0)

    def test_gap_or_time_cap_keeps_baseline(self):
        self.video = Video(self.path, 1920, 1088, tuple(frame / 30 for frame in range(25)))
        fixture = self.fixture(lambda frame, t: (100, 150), [candidate(200)], winner=0)
        _, row = self.replay(fixture, "reject")
        self.assertEqual(row["check"], "inconclusive")
        self.assertEqual(row["selected_index"], 0)

        settings = MotionSettings(mask_radius=1)
        fixture = self.fixture(
            lambda frame, t: (100, 150), [candidate(200)], winner=0, gaps=range(4, 11),
        )
        _, row = self.replay(fixture, "reject", settings)
        self.assertEqual(row["check"], "inconclusive")
        self.assertEqual(row["selected_index"], 0)

    def test_only_rerank_can_fill_blank_with_a_previously_gated_candidate(self):
        fixture = self.fixture(
            lambda frame, t: (100, 150), [candidate(100, passed_gate=False)], winner=None,
        )
        _, rejected = self.replay(fixture, "reject")
        _, reranked = self.replay(fixture, "rerank")
        self.assertIsNone(rejected["selected_index"])
        self.assertEqual(reranked["selected_index"], 0)

    def test_rejects_mismatched_winner(self):
        fixture = self.fixture(lambda frame, t: (100, 150), [candidate(100)], winner=0)
        fixture[1]["frames"][0]["candidates"][0]["x_px"] = 110
        with self.assertRaisesRegex(ValueError, "logged winner"):
            self.replay(fixture, "rerank")

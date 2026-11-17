import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

from snufspin.tennis_vision import import_ball_track
from snufspin.tracking import BallPoint, file_sha256, read_ball_track
from snufspin.video import Video


@unittest.skipIf(np is None, "the tennis-vision adapter requires NumPy")
class TennisVisionImportTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.artifact = Path(directory.name) / "upstream.npz"
        source = Path(directory.name) / "clip.mp4"
        source.write_bytes(b"source video")
        self.video = Video(source, 320, 180, (0.0, 0.1, 0.4))

    def save_artifact(self, **overrides):
        fields = {
            "video_sha256": file_sha256(self.video.path),
            "frame_count": 3,
            "fps": 30.0,
            "ball_track": np.array([[0, 20], [np.nan, np.nan], [31.5, 40]], dtype=np.float32),
            "kps_court": np.array([None, None, None], dtype=object),
        }
        np.savez_compressed(self.artifact, **(fields | overrides))

    def test_import_uses_video_timing_and_preserves_missing_detections(self):
        self.save_artifact()
        report = import_ball_track(self.video, self.artifact)
        self.assertEqual(report["timestamps_s"], (0.0, 0.1, 0.4))
        self.assertEqual(read_ball_track(self.video, report), (
            BallPoint(0, 0, 20, "measured"),
            BallPoint(2, 31.5, 40, "measured"),
        ))

    def test_rejects_mismatched_source_video(self):
        self.save_artifact(video_sha256="other recording")
        with self.assertRaisesRegex(ValueError, "different video"):
            import_ball_track(self.video, self.artifact)

    def test_rejects_invalid_or_pickled_ball_tracks(self):
        for track in (
            np.array([[0.0, 20], [np.nan, 20], [30, 40]]),
            np.array([[0.0, 20], [None, None], [30, 40]], dtype=object),
        ):
            with self.subTest(dtype=track.dtype):
                self.save_artifact(ball_track=track)
                with self.assertRaises(ValueError):
                    import_ball_track(self.video, self.artifact)

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from snufspin.overlay import render_overlay
from snufspin.tracking import BallPoint, read_ball_track, track_report
from snufspin.video import Video, probe_video


class TrackingTests(unittest.TestCase):
    def test_cached_tracks_are_bound_to_the_input_video(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            path.write_bytes(b"first recording")
            video = Video(path, 320, 180, (0.0, 0.1, 0.4))
            points = (BallPoint(1, 30, 40, "measured"),)
            report = json.loads(json.dumps(track_report(video, points, {"engine": "test"})))
            self.assertEqual(read_ball_track(video, report), points)
            path.write_bytes(b"another recording")
            with self.assertRaisesRegex(ValueError, "different video"):
                read_ball_track(video, report)

    def test_overlay_preserves_frame_timing_and_hides_marker_on_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            output = Path(directory) / "overlay.mp4"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=320x180:r=30:d=1",
                "-vf", "select='eq(n,0)+eq(n,3)+eq(n,12)'", "-fps_mode", "vfr", str(source),
            ], check=True)
            video = probe_video(source)
            render_overlay(video, (BallPoint(1, 50, 50, "measured"),), output)
            self.assertEqual(probe_video(output).timestamps_s, (0.0, 0.1, 0.4))
            frames = subprocess.check_output([
                "ffmpeg", "-v", "error", "-i", str(output), "-fps_mode", "passthrough",
                "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
            ])
            green_channel = ((45 * 320) + 50) * 3 + 1
            frame_size = 320 * 180 * 3
            self.assertEqual(frames[green_channel], 0)
            self.assertGreater(frames[frame_size + green_channel], 150)
            self.assertEqual(frames[frame_size * 2 + green_channel], 0)

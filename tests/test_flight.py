import unittest
from pathlib import Path

import numpy as np

from snufspin.calibration import Camera, Intrinsics
from snufspin.flight import fit_flight
from snufspin.tracking import BallPoint
from snufspin.video import Video


class FlightFitTests(unittest.TestCase):
    def setUp(self):
        self.video = Video(Path("video.mp4"), 1920, 1080,
                           tuple(np.cumsum([0, *([0.016, 0.017, 0.019] * 12)])))
        center = np.array([-8., 11., 3.])
        forward = np.array([9., 5., 0.]) - center
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        R = np.vstack((right, np.cross(forward, right), forward))
        self.camera = Camera(Intrinsics([[1450, 0, 960], [0, 1450, 540], [0, 0, 1]],
                                        [-0.04, 0.01, 0, 0]), R, -R @ center)
        self.xyz = np.array([
            [6 + 12 * t, 6 - 3 * t, 1.8 + 3.2 * t - 0.5 * 9.80665 * t**2]
            for t in self.video.timestamps_s
        ])
        self.pixels = self.camera.project(self.xyz)

    def test_recovers_known_3d_path_with_outliers_and_irregular_timestamps(self):
        points = []
        for frame, (x, y) in enumerate(self.pixels):
            provenance = "predicted" if frame == 17 else "measured"
            if frame in (8, 17, 22):
                x += 90
                y -= 45
            points.append(BallPoint(frame, x, y, provenance))
        fit = fit_flight(self.video, self.camera, tuple(points), 0, 36)
        self.assertEqual(fit.diagnostics["status"], "accepted", fit.diagnostics)
        self.assertEqual(fit.diagnostics["excluded_inferred_frames"], [17])
        self.assertNotIn(8, fit.diagnostics["inlier_frames"])
        self.assertNotIn(22, fit.diagnostics["inlier_frames"])
        np.testing.assert_allclose(fit.diagnostics["velocity_at_first_sample_m_s"], [12, -3, 3.2], atol=1e-5)
        np.testing.assert_allclose(
            [(point.x_m, point.y_m, point.z_m) for point in fit.track.samples],
            self.xyz, atol=1e-5,
        )

    def test_excludes_interpolated_observations_from_the_required_support(self):
        points = tuple(BallPoint(frame, x, y, "measured" if frame < 4 else "interpolated")
                       for frame, (x, y) in enumerate(self.pixels))
        result = fit_flight(self.video, self.camera, points, 0, 36)
        self.assertEqual(result.diagnostics["reason"], "insufficient_measured_points")
        self.assertIsNone(result.track)

    def test_does_not_extrapolate_to_an_unobserved_window_endpoint(self):
        points = tuple(BallPoint(frame, *self.pixels[frame], "manual") for frame in range(3, 32))
        result = fit_flight(self.video, self.camera, points, 0, 36)
        self.assertIsNotNone(result.track)
        self.assertEqual((result.track.samples[0].frame, result.track.samples[-1].frame), (3, 31))

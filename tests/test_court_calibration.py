import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from snufspin.calibration import Calibration, Camera, CourtMapping, Intrinsics, read_calibration
from snufspin.court import LANDMARKS, LINES
from snufspin.court_fit import estimate_intrinsics, fit_court
from snufspin.video import Video


def camera_fixture(center=(-12, 13, 9), distortion=None):
    center = np.array(center, dtype=float)
    forward = np.array([11.885, 5.485, 0]) - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    R = np.vstack((right, np.cross(forward, right), forward))
    return Camera(Intrinsics([[800, 0, 639.5], [0, 800, 359.5], [0, 0, 1]], distortion),
                  R, -R @ center)


def frame_fixture(camera):
    image = np.full((720, 1280, 3), (58, 130, 76), np.uint8)
    points = {name: camera.project(xyz) for name, xyz in LANDMARKS.items()}
    for start, end in LINES.values():
        xyz = np.linspace(LANDMARKS[start], LANDMARKS[end], 100)
        uv = np.rint(camera.project(xyz)).astype(np.int32)
        cv2.polylines(image, [uv], False, (249, 249, 249), 3, cv2.LINE_AA)
    return image, points


class CourtCalibrationTests(unittest.TestCase):
    def test_distorted_projection_ray_and_court_mapping(self):
        camera = camera_fixture(distortion=[-0.09, 0.02, 0.001, 0.002, 0])
        point = np.array([8, 9, 3])
        origin, direction = camera.pixel_ray(camera.project(point))
        np.testing.assert_allclose(origin + np.linalg.norm(point - origin) * direction, point, atol=1e-8)
        mapping = CourtMapping.from_camera(camera)
        ground = np.array([7.4, 8.5, 0])
        np.testing.assert_allclose(mapping.pixel_to_court(camera.project(ground)), ground, atol=1e-8)
        with self.assertRaises(ValueError):
            camera.project(camera.center - camera.R[2])

    def test_video_bound_sidecar_roundtrip(self):
        camera = camera_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "video"
            path.write_bytes(b"video")
            video = Video(path, 1280, 720, (0, 1, 2))
            calibration = Calibration(CourtMapping.from_camera(camera), {"method": "test"}, camera)
            report = calibration.to_report(video)
            loaded = read_calibration(video, report)
            np.testing.assert_allclose(loaded.camera.center, camera.center)
            path.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "different video"):
                read_calibration(video, report)

    def test_fit_known_intrinsics_and_estimated_focal_with_outlier(self):
        for distortion in (None, [-0.04, 0.01, 0.002, 0.002, 0]):
            camera = camera_fixture(distortion=distortion)
            frame, points = frame_fixture(camera)
            points["service_b_center"] += [36, -28]
            landmarks = {0: points, 1: points}
            K = camera.intrinsics if distortion else None
            calibration, report = fit_court({0: frame, 1: frame}, landmarks, {}, K)
            self.assertEqual(report["status"], "camera", report)
            self.assertNotIn("service_b_center", report["frames"]["0"]["inliers"])
            np.testing.assert_allclose(calibration.camera.center, camera.center, atol=1e-4)
            np.testing.assert_allclose(calibration.camera.intrinsics.K, camera.intrinsics.K, atol=0.01)

    def test_court_only_when_vanishing_point_is_at_infinity(self):
        camera = camera_fixture(center=(-18, 5.485, 10))
        frame, points = frame_fixture(camera)
        self.assertIsNone(estimate_intrinsics(CourtMapping.from_camera(camera), 1280, 720))
        calibration, report = fit_court({0: frame, 1: frame}, {0: points, 1: points}, {})
        self.assertEqual(report["status"], "court_mapping_only")
        self.assertIsNone(calibration.camera)
        np.testing.assert_allclose(
            calibration.mapping.pixel_to_court(points["service_a_center"]),
            LANDMARKS["service_a_center"], atol=1e-5,
        )

    def test_line_validation_rejects_blank_frames_and_camera_motion(self):
        frame, points = frame_fixture(camera_fixture())
        other, other_points = frame_fixture(camera_fixture(center=(-14, 10, 9)))
        calibration, report = fit_court(
            {0: frame, 1: other}, {0: points, 1: other_points}, {},
        )
        self.assertIsNone(calibration)
        self.assertEqual(report["status"], "inconsistent_court_across_frames")
        blank = np.zeros_like(frame)
        calibration, report = fit_court({0: frame, 1: blank}, {0: points}, {})
        self.assertIsNone(calibration)
        self.assertEqual(report["status"], "shared_homography_failed_marking_check")

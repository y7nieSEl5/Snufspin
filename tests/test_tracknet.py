import importlib.util
import unittest
from unittest.mock import patch

from snufspin.tracknet import select_ball


@unittest.skipUnless(
    importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"),
    "detector post-processing requires OpenCV and NumPy",
)
class TrackNetPostprocessTests(unittest.TestCase):
    def test_preserves_baseline_mask_conversion(self):
        import numpy as np

        heat = np.array([[0, 1, 15, 128, 255]], dtype=np.int64)
        with patch("cv2.HoughCircles", return_value=None) as circles:
            self.assertIsNone(select_ball(heat, None, 2, 2))
        np.testing.assert_array_equal(circles.call_args.args[0], [[0, 255, 255, 255, 0]])
        np.testing.assert_array_equal(heat, [[0, 1, 15, 128, 255]])

    def test_scales_coordinates_and_checks_candidates_in_order(self):
        import numpy as np

        candidates = np.array([[[150, 100, 3], [30, 40, 3]]], dtype=np.float32)
        heat = np.zeros((360, 640), dtype=np.int64)
        with patch("cv2.HoughCircles", return_value=candidates):
            self.assertEqual(select_ball(heat, None, 2, 2), (300, 200))
            self.assertEqual(select_ball(heat, (50, 70), 2, 2), (60, 80))
            self.assertIsNone(select_ball(heat, (500, 500), 2, 2))

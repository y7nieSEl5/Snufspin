import importlib.util
import unittest

import cv2
import numpy as np

from snufspin.calibration import CourtMapping
from snufspin.court import LANDMARKS, LINES


@unittest.skipUnless(importlib.util.find_spec("scipy"), "optional TEXflip dependencies are required")
class CourtPaintTests(unittest.TestCase):
    def frame_fixture(self):
        mapping = CourtMapping([[20, -110, 1020], [-3, 0, 590], [0.05, 0, 1]])
        image = np.full((720, 1280), 75, dtype=np.uint8)
        segments = np.array([mapping.project([LANDMARKS[n][:2] for n in ends]) for ends in LINES.values()])
        for a, b in np.rint(segments).astype(int):
            cv2.line(image, a, b, 240, 5)
        return mapping, image, segments

    def test_paint_prefers_the_alignment_to_a_vertical_shift(self):
        from snufspin.court_paint import score_mapping

        mapping, image, _ = self.frame_fixture()
        shifted = CourtMapping([[1, 0, 0], [0, 1, 15], [0, 0, 1]] @ mapping.H)
        correct = score_mapping(mapping, {0: image, 1: image})
        displaced = score_mapping(shifted, {0: image, 1: image})
        self.assertTrue(correct.frames[0]["supported"])
        self.assertGreater(correct.score, displaced.score)

    def test_offscreen_segment_contributes_zero(self):
        from snufspin.court_paint import _sample_segment

        points, _, visible = _sample_segment(np.array([[-10., 7.], [-3., 19.]]), 1280, 720)
        self.assertEqual(visible, 0)
        self.assertEqual(len(points), 0)

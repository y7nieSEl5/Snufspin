import importlib.util
import unittest

import numpy as np

from snufspin.calibration import CourtMapping
from snufspin.court import LANDMARKS, LINES


@unittest.skipUnless(importlib.util.find_spec("scipy"), "optional TEXflip dependencies are required")
class LineCourtTests(unittest.TestCase):
    def test_clipped_segments_yield_named_observed_intersections(self):
        from snufspin.line_court import fit_lines

        # Near doubles corners are outside the frame. Partial sidelines still constrain the fit.
        mapping = CourtMapping([[45, -240, 2200], [2, 0, 900], [0.06, 0, 1]])
        segments = []
        for ends in LINES.values():
            a, b = mapping.project([LANDMARKS[name][:2] for name in ends])
            dx = b[0] - a[0]
            bounds = sorted(((0 - a[0]) / dx, (1919 - a[0]) / dx))
            segments.append([a + max(0, bounds[0]) * (b - a), a + min(1, bounds[1]) * (b - a)])
        points = fit_lines(np.array(segments), 1920, 1080)
        self.assertGreaterEqual(len(points), 10)
        self.assertNotIn("baseline_a_doubles_low", points)
        self.assertNotIn("baseline_a_doubles_high", points)
        for name, point in points.items():
            np.testing.assert_allclose(point, mapping.project(LANDMARKS[name][:2]), atol=1e-5)

    def test_parallel_segments_cannot_produce_points(self):
        from snufspin.line_court import fit_lines

        segments = np.array([[[100, y], [500, y]] for y in range(100, 500, 40)])
        self.assertEqual(fit_lines(segments, 1920, 1080), {})

    def test_refinement_centers_a_segment_on_paint(self):
        import cv2

        from snufspin.line_court import refine_segments

        frame = np.full((720, 1280, 3), 70, dtype=np.uint8)
        cv2.line(frame, (100, 445), (1100, 445), (250, 250, 250), 5)
        segments = np.array([[[120, 455], [1080, 455]], [[120, 100], [1080, 100]]], dtype=float)
        refined = refine_segments(frame, segments)
        np.testing.assert_allclose(refined[0, :, 1], 445, atol=2)
        np.testing.assert_array_equal(refined[1], segments[1])

import unittest

from snufspin.wasb import _selected_frames


class CandidateRecorder:
    def update(self, candidates):
        return list(candidates)


class WasbWindowTest(unittest.TestCase):
    def test_overlap_preserves_frame_indexes_and_flushes_tail(self):
        def detect(window):
            return [
                [] if frame == 2 else [(window[0], frame)]
                for frame in window
            ]

        frames = list(_selected_frames(range(5), 3, detect, CandidateRecorder()))
        self.assertEqual(frames, [
            (0, [(0, 0)]),
            (1, [(0, 1), (1, 1)]),
            (2, []),
            (3, [(1, 3), (2, 3)]),
            (4, [(2, 4)]),
        ])

    def test_short_clip_preserves_both_frame_indexes(self):
        def detect(window):
            self.fail("a two-frame clip has no complete window")

        frames = list(_selected_frames(range(2), 3, detect, CandidateRecorder()))
        self.assertEqual(frames, [(0, []), (1, [])])

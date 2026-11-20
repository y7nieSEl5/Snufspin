import unittest

from snufspin.wasb import WasbCandidateLog, _selected_frames


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

    def test_stride_three_uses_disjoint_windows_and_preserves_uncovered_tail(self):
        def detect(window):
            return [[(window[0], frame)] for frame in window]

        frames = list(_selected_frames(range(7), 3, detect, CandidateRecorder(), stride=3))
        self.assertEqual(frames, [
            (0, [(0, 0)]),
            (1, [(0, 1)]),
            (2, [(0, 2)]),
            (3, [(3, 3)]),
            (4, [(3, 4)]),
            (5, [(3, 5)]),
            (6, []),
        ])

    def test_candidate_log_observes_gate_and_first_winner_without_changing_track(self):
        class Tracker:
            def _select_not_too_far(self, candidates):
                return [candidate for candidate in candidates if candidate["xy"][0] < 100]

            def update(self, candidates):
                passed = self._select_not_too_far(candidates)
                if not passed:
                    return {"visi": False}
                selected = max(passed, key=lambda candidate: candidate["score"])
                x, y = selected["xy"]
                return {"visi": True, "x": x, "y": y, "score": selected["score"]}

        candidates = [
            {"xy": (150.0, 2.0), "score": 10.0},
            {"xy": (50.0, 2.0), "score": 2.0},
            {"xy": (50.0, 2.0), "score": 2.0},
        ]

        def detect(window):
            return [candidates, candidates, []]

        log = WasbCandidateLog(1, 2)
        plain = list(_selected_frames(range(3), 3, detect, Tracker()))
        observed = list(_selected_frames(range(3), 3, detect, Tracker(), candidate_log=log))
        self.assertEqual(observed, plain)
        self.assertEqual([item["frame"] for item in log.frames], [1, 2])
        self.assertEqual(
            [entry["passed_gate"] for entry in log.frames[0]["candidates"]], [False, True, True],
        )
        self.assertEqual(log.frames[0]["selected_index"], 1)
        self.assertIsNone(log.frames[1]["selected_index"])

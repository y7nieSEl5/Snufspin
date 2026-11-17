"""An opt-in short-gap interpolation experiment over cached observations."""

from bisect import bisect_right
from dataclasses import asdict
from math import isfinite

from .tracking import BallPoint, index_ball_points, read_ball_track
from .video import Video


def interpolate_short_gaps(video: Video, report: dict, max_gap_s: float = 0.15) -> dict:
    if not isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be finite and positive")
    points = read_ball_track(video, report)
    by_frame = index_ball_points(video, points)
    measured = sorted(
        (point for point in points if point.provenance in {"measured", "manual"}),
        key=lambda point: point.frame,
    )
    occupied = sorted(by_frame)
    for before, after in zip(measured, measured[1:]):
        duration = video.time_at(after.frame) - video.time_at(before.frame)
        if after.frame == before.frame + 1 or duration > max_gap_s + 1e-9:
            continue
        # Preserve existing inferred samples and do not bridge through them.
        if occupied[bisect_right(occupied, before.frame)] != after.frame:
            continue
        for frame in range(before.frame + 1, after.frame):
            fraction = (video.time_at(frame) - video.time_at(before.frame)) / duration
            by_frame[frame] = BallPoint(
                frame,
                before.x_px + fraction * (after.x_px - before.x_px),
                before.y_px + fraction * (after.y_px - before.y_px),
                "interpolated",
            )
    return report | {
        "source": report["source"] | {"experiment": "short_gap_interpolation", "max_gap_s": max_gap_s},
        "ball_track": [asdict(by_frame[frame]) for frame in sorted(by_frame)],
    }

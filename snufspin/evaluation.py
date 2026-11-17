"""Evaluate a cached ball track on the frames a reviewer explicitly labeled."""

from math import hypot, isfinite

from .tracking import read_ball_track
from .video import Video


def evaluate_review(video: Video, report: dict, review: dict) -> dict:
    points = {point.frame: point for point in read_ball_track(video, report)}
    if review["video_sha256"] != report["video_sha256"]:
        raise ValueError("review was made for a different video")
    labels = review["corrections"]
    if not labels:
        raise ValueError("review must contain at least one labeled frame")

    frames = []
    seen = set()
    for label in labels:
        frame = label["frame"]
        time_s = video.time_at(frame)
        if frame in seen:
            raise ValueError(f"duplicate review frame {frame}")
        seen.add(frame)
        timestamp = label["timestamp_s"]
        if not isfinite(timestamp) or abs(timestamp - time_s) > 1e-6:
            raise ValueError(f"review timestamp does not match frame {frame}")
        status = label["status"]
        if status not in {"visible", "hidden"}:
            raise ValueError(f"unknown review status at frame {frame}")
        point = points.get(frame)
        error = None
        if status == "visible":
            x, y = label["x_px"], label["y_px"]
            if not (0 <= x < video.width and 0 <= y < video.height):
                raise ValueError(f"review coordinates outside image at frame {frame}")
            if point is not None:
                error = hypot(point.x_px - x, point.y_px - y)
        frames.append({
            "frame": frame,
            "timestamp_s": time_s,
            "label": status,
            "has_position": point is not None,
            "provenance": point.provenance if point else None,
            "error_px": error,
        })

    visible = [frame for frame in frames if frame["label"] == "visible"]
    hidden = [frame for frame in frames if frame["label"] == "hidden"]
    errors = [frame["error_px"] for frame in visible if frame["error_px"] is not None]
    return {
        "video_sha256": report["video_sha256"],
        "scope": "manually_selected_review_frames",
        "reviewed_frames": len(frames),
        "visible_frames": len(visible),
        "hidden_frames": len(hidden),
        "visible_without_position": sum(not frame["has_position"] for frame in visible),
        "hidden_with_position": sum(frame["has_position"] for frame in hidden),
        "localized_within_px": {
            str(tolerance): sum(error <= tolerance for error in errors)
            for tolerance in (5, 10, 20)
        },
        "mean_error_px_when_position_present": sum(errors) / len(errors) if errors else None,
        "frames": sorted(frames, key=lambda frame: frame["frame"]),
    }

"""The frame-indexed ball observations shared by tracking and analysis."""

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from .video import Video


@dataclass(frozen=True)
class BallPoint:
    frame: int
    x_px: float
    y_px: float
    provenance: str
    confidence: float | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_ball_points(video: Video, points: tuple[BallPoint, ...]) -> dict[int, BallPoint]:
    by_frame = {}
    for point in points:
        video.time_at(point.frame)
        if point.frame in by_frame:
            raise ValueError(f"multiple ball points for frame {point.frame}")
        if not (0 <= point.x_px < video.width and 0 <= point.y_px < video.height):
            raise ValueError(f"ball point outside the image at frame {point.frame}")
        if point.provenance not in {"manual", "measured", "interpolated", "predicted"}:
            raise ValueError(f"invalid ball point provenance: {point.provenance!r}")
        if point.confidence is not None and not 0 <= point.confidence <= 1:
            raise ValueError(f"invalid ball point confidence at frame {point.frame}")
        by_frame[point.frame] = point
    return by_frame


def track_report(video: Video, points: tuple[BallPoint, ...], source: dict[str, str]) -> dict:
    by_frame = index_ball_points(video, points)
    return {
        "schema_version": 1,
        "video_sha256": file_sha256(video.path),
        "width": video.width,
        "height": video.height,
        "frame_count": len(video.timestamps_s),
        "timestamps_s": video.timestamps_s,
        "source": source,
        "ball_track": [asdict(by_frame[frame]) for frame in sorted(by_frame)],
    }


def read_ball_track(video: Video, report: dict) -> tuple[BallPoint, ...]:
    if report["schema_version"] != 1:
        raise ValueError("unsupported ball track schema version")
    if report["video_sha256"] != file_sha256(video.path):
        raise ValueError("ball track was generated for a different video")
    if (
        report["width"] != video.width
        or report["height"] != video.height
        or report["frame_count"] != len(video.timestamps_s)
        or tuple(report["timestamps_s"]) != video.timestamps_s
    ):
        raise ValueError("ball track frame metadata does not match the video")
    points = tuple(BallPoint(**point) for point in report["ball_track"])
    index_ball_points(video, points)
    return points

from dataclasses import asdict, dataclass
from math import dist, isfinite
from typing import Callable

from .video import Video


@dataclass(frozen=True)
class BallPoint:
    frame: int
    x_px: float
    y_px: float
    provenance: str
    confidence: float | None = None


@dataclass(frozen=True)
class ShotLabels:
    stroke: str | None = None
    intent: str | None = None
    spin: tuple[str, ...] | None = None
    source: str = "manual"


@dataclass(frozen=True)
class MetricPoint:
    frame: int
    x_m: float
    y_m: float
    z_m: float


@dataclass(frozen=True)
class MetricTrack:
    method: str
    samples: tuple[MetricPoint, ...]


@dataclass(frozen=True)
class Shot:
    contact_frame: int
    end_frame: int
    end_event: str | None = None
    labels: ShotLabels | None = None
    metric_track: MetricTrack | None = None


@dataclass(frozen=True)
class ShotInput:
    video: Video
    shot: Shot
    ball_track: tuple[BallPoint, ...]


Classifier = Callable[[ShotInput], ShotLabels | None]
Reconstructor = Callable[[ShotInput], MetricTrack | None]


def supplied_labels(inputs: ShotInput) -> ShotLabels | None:
    return inputs.shot.labels


def supplied_metric_track(inputs: ShotInput) -> MetricTrack | None:
    return inputs.shot.metric_track


def summarize_speed(video: Video, shot: Shot, track: MetricTrack | None) -> dict:
    if track is None or not track.samples:
        return {"status": "unavailable", "reason": "missing_metric_trajectory"}

    frames = [point.frame for point in track.samples]
    times = [video.time_at(frame) for frame in frames]
    if any(a >= b for a, b in zip(frames, frames[1:])):
        raise ValueError("metric trajectory frames must be unique and increasing")
    if frames[0] < shot.contact_frame or frames[-1] > shot.end_frame:
        raise ValueError("metric trajectory samples must lie within the shot")

    coordinates = [(point.x_m, point.y_m, point.z_m) for point in track.samples]
    if not all(isfinite(axis) for point in coordinates for axis in point):
        raise ValueError("metric trajectory coordinates must be finite")
    if len(times) < 2:
        return {"status": "unavailable", "reason": "insufficient_metric_samples"}
    distance_m = sum(dist(a, b) for a, b in zip(coordinates, coordinates[1:]))
    average_speed_kmh = 3.6 * distance_m / (times[-1] - times[0])
    if not isfinite(average_speed_kmh):
        raise ValueError("metric trajectory produced a non-finite speed")

    return {
        "status": "available",
        "method": track.method,
        "average_speed_kmh": average_speed_kmh,
        "start_s": times[0],
        "end_s": times[-1],
        "samples": len(times),
        "covers_shot": frames[0] == shot.contact_frame and frames[-1] == shot.end_frame,
    }


def analyze(
    video: Video,
    ball_track: tuple[BallPoint, ...],
    shots: tuple[Shot, ...],
    *,
    classify: Classifier = supplied_labels,
    reconstruct: Reconstructor = supplied_metric_track,
) -> list[dict]:
    points_by_frame = {}
    for point in ball_track:
        video.time_at(point.frame)
        if point.frame in points_by_frame:
            raise ValueError(f"multiple ball points for frame {point.frame}")
        if not (0 <= point.x_px < video.width and 0 <= point.y_px < video.height):
            raise ValueError(f"ball point outside the image at frame {point.frame}")
        if point.provenance not in {"manual", "measured", "interpolated", "predicted"}:
            raise ValueError(f"invalid ball point provenance: {point.provenance!r}")
        if point.confidence is not None and not 0 <= point.confidence <= 1:
            raise ValueError(f"invalid ball point confidence at frame {point.frame}")
        points_by_frame[point.frame] = point

    results = []
    previous_end = -1
    for shot in shots:
        contact_s = video.time_at(shot.contact_frame)
        end_s = video.time_at(shot.end_frame)
        if shot.contact_frame < previous_end or shot.end_frame <= shot.contact_frame:
            raise ValueError("shots must be ordered and may only overlap at a shared endpoint")
        previous_end = shot.end_frame
        points = tuple(
            points_by_frame[frame]
            for frame in range(shot.contact_frame, shot.end_frame + 1)
            if frame in points_by_frame
        )
        inputs = ShotInput(video, shot, points)
        labels = classify(inputs)
        results.append({
            "contact_frame": shot.contact_frame,
            "contact_s": contact_s,
            "end_frame": shot.end_frame,
            "end_s": end_s,
            "end_event": shot.end_event,
            "tracked_frames": len(points),
            "labels": asdict(labels) if labels is not None else None,
            "speed": summarize_speed(video, shot, reconstruct(inputs)),
        })
    return results

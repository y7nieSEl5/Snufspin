"""Replay a frozen WASB candidate log with a two-sided image motion check."""

from dataclasses import asdict, dataclass
from itertools import combinations
from math import dist, isfinite

from .tracking import BallPoint, read_ball_track
from .video import Video


@dataclass(frozen=True)
class MotionSettings:
    mask_radius: int = 3
    side_frames: int = 6
    tolerance_heatmap_px: float = 3.0
    max_extrapolation_s: float = 0.1


@dataclass(frozen=True)
class Line:
    # Time is relative to the target frame for both lines.
    center: tuple[float, float]
    velocity: tuple[float, float]
    support_times: tuple[float, ...]

    def at(self, time):
        return tuple(center + speed * time for center, speed in zip(self.center, self.velocity))


def _fit_line(samples, tolerance):
    """Find the pair with largest consensus, then refit on its inliers."""
    best = ()
    best_key = (0, 0.0)
    for (start, a), (end, b) in combinations(samples, 2):
        velocity = tuple((right - left) / (end - start) for left, right in zip(a, b))
        errors = []
        inliers = []
        for time, position in samples:
            predicted = tuple(x + speed * (time - start) for x, speed in zip(a, velocity))
            error = dist(position, predicted)
            if error <= tolerance:
                inliers.append((time, position))
                errors.append(error ** 2)
        key = (len(inliers), -sum(errors))
        if key > best_key:
            best, best_key = inliers, key
    if len(best) < 3:
        return None

    times = tuple(time for time, _ in best)
    mean_time = sum(times) / len(times)
    variance = sum((time - mean_time) ** 2 for time in times)
    positions = tuple(zip(*(point for _, point in best)))
    means = tuple(sum(axis) / len(axis) for axis in positions)
    velocity = tuple(
        sum((time - mean_time) * (value - mean) for time, value in zip(times, axis)) / variance
        for mean, axis in zip(means, positions)
    )
    return Line(tuple(mean - speed * mean_time for mean, speed in zip(means, velocity)), velocity, times)


def _motion_check(video, winners, frame, settings, tolerance):
    target_time = video.time_at(frame)
    first = max(0, frame - settings.mask_radius)
    last = min(len(video.timestamps_s) - 1, frame + settings.mask_radius)
    frame_ranges = (
        range(max(0, first - settings.side_frames), first),
        range(last + 1, min(len(video.timestamps_s), last + 1 + settings.side_frames)),
    )
    lines = []
    for frame_range in frame_ranges:
        samples = [(video.time_at(i) - target_time, winners[i]) for i in frame_range if i in winners]
        line = _fit_line(samples, tolerance)
        if line is None or min(map(abs, line.support_times)) > settings.max_extrapolation_s + 1e-9:
            return "inconclusive", (), None
        lines.append(line)

    incoming, outgoing = lines
    if dist(incoming.center, outgoing.center) <= tolerance:
        return "straight", (incoming.center, outgoing.center), None

    # Minimize squared separation with a shared time, within the masked
    # window where both fits are still close enough to their own support.
    horizon = settings.max_extrapolation_s
    earliest = max(video.time_at(first) - target_time, outgoing.support_times[0] - horizon)
    latest = min(video.time_at(last) - target_time, incoming.support_times[-1] + horizon)
    delta = tuple(a - b for a, b in zip(incoming.center, outgoing.center))
    relative_velocity = tuple(a - b for a, b in zip(incoming.velocity, outgoing.velocity))
    speed_squared = sum(speed ** 2 for speed in relative_velocity)
    if speed_squared == 0 or earliest > latest:
        return "inconclusive", (), None
    closest = -sum(d * speed for d, speed in zip(delta, relative_velocity)) / speed_squared
    contact = min(max(closest, earliest), latest)
    if dist(incoming.at(contact), outgoing.at(contact)) > tolerance:
        return "inconclusive", (), None
    if abs(contact) <= 1e-9:
        predictions = (incoming.center, outgoing.center)
    else:
        predictions = ((incoming if contact > 0 else outgoing).center,)
    return "bend", predictions, target_time + contact


def replay_wasb(
    video: Video, baseline: dict, log: dict, mode: str, settings: MotionSettings = MotionSettings(),
) -> tuple[dict, list[dict]]:
    """Return a track and per-frame decisions, using only frozen winners as fit evidence."""
    if mode not in ("reject", "rerank"):
        raise ValueError("motion mode must be reject or rerank")
    if settings.mask_radius < 0 or settings.side_frames < 3 or any(
        not isfinite(value) or value <= 0
        for value in (settings.tolerance_heatmap_px, settings.max_extrapolation_s)
    ):
        raise ValueError("invalid motion settings")

    points = {point.frame: point for point in read_ball_track(video, baseline)}
    for key in ("schema_version", "video_sha256", "width", "height", "frame_count", "source"):
        if log[key] != baseline[key]:
            raise ValueError(f"candidate log and baseline differ in {key}")
    if tuple(log["timestamps_s"]) != video.timestamps_s:
        raise ValueError("candidate log timestamps do not match the video")
    source = baseline["source"]
    if source["engine"] != "wasb-tennis" or source["heatmap_scale"] != 0 or source["input_width"] <= 0:
        raise ValueError("expected a WASB Tennis scale-0 candidate log")
    if any(point.provenance != "measured" for point in points.values()):
        raise ValueError("motion replay requires a measured baseline")

    # WASB's scale-0 affine warp has an isotropic scale, even when the
    # source aspect ratio differs from the heatmap's aspect ratio.
    source_pixels_per_heatmap_pixel = max(video.width, video.height) / source["input_width"]
    tolerance = settings.tolerance_heatmap_px * source_pixels_per_heatmap_pixel
    score_tie = 0.1 * source_pixels_per_heatmap_pixel
    winners = {frame: (point.x_px, point.y_px) for frame, point in points.items()}
    decisions = []
    seen = set()
    for record in log["frames"]:
        frame = record["frame"]
        time = video.time_at(frame)
        if frame in seen or abs(time - record["timestamp_s"]) > 1e-6:
            raise ValueError(f"duplicate frame or mismatched timestamp in candidate log at {frame}")
        seen.add(frame)
        candidates = record["candidates"]
        positions = [(c["x_px"], c["y_px"]) for c in candidates]
        eligible = [i for i, (x, y) in enumerate(positions) if 0 <= x < video.width and 0 <= y < video.height]
        winner = record["selected_index"]
        if winner is not None and (type(winner) is not int or not 0 <= winner < len(candidates)):
            raise ValueError(f"invalid winner index at frame {frame}")
        baseline_index = winner if winner in eligible else None
        if (positions[baseline_index] if baseline_index is not None else None) != winners.get(frame):
            raise ValueError(f"logged winner does not match the baseline at frame {frame}")

        check, predictions, contact = _motion_check(video, winners, frame, settings, tolerance)
        errors = [None] * len(candidates)
        if predictions:
            for i in eligible:
                errors[i] = max(dist(positions[i], point) for point in predictions)

        chosen = baseline_index
        if predictions and (baseline_index is None or errors[baseline_index] > tolerance):
            chosen = None
            if mode == "rerank":
                supported = [i for i in eligible if errors[i] <= tolerance]
                if supported:
                    best = min(errors[i] for i in supported)
                    close = [i for i in supported if errors[i] <= best + score_tie]
                    chosen = min(close, key=lambda i: (-candidates[i]["score"], errors[i], i))
        if chosen != baseline_index:
            if chosen is None:
                points.pop(frame)
            else:
                points[frame] = BallPoint(frame, *positions[chosen], "measured")
        decisions.append({
            "frame": frame, "timestamp_s": time, "check": check, "contact_time_s": contact,
            "predictions_px": predictions, "candidate_errors_px": errors,
            "baseline_index": baseline_index, "selected_index": chosen,
        })

    report = baseline | {
        "source": source | {
            "experiment": "wasb_motion_replay", "mode": mode, **asdict(settings),
            "tolerance_source_px": tolerance, "score_tie_heatmap_px": 0.1,
        },
        "ball_track": [asdict(points[frame]) for frame in sorted(points)],
    }
    return report, decisions

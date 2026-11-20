"""A calibrated, gravity-only fit for one manually delimited airborne interval."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares

from .calibration import Camera
from .pipeline import MetricPoint, MetricTrack
from .tracking import BallPoint, index_ball_points
from .video import Video

_GRAVITY = np.array([0.0, 0.0, -9.80665])
_MIN_POINTS = 6
_MIN_SPAN_S = 0.12


@dataclass(frozen=True)
class FlightFit:
    track: MetricTrack | None
    diagnostics: dict


def _trajectory(parameters, times, origin_s, scale_s):
    origin = parameters[:3]
    velocity = parameters[3:] / scale_s
    elapsed = times - origin_s
    return origin + elapsed[:, None] * velocity + 0.5 * elapsed[:, None] ** 2 * _GRAVITY


def fit_flight(
    video: Video, camera: Camera, points: tuple[BallPoint, ...], start_frame: int, end_frame: int,
) -> FlightFit:
    """Fit one free-flight arc to measured source pixels using video timestamps."""
    start_s, end_s = video.time_at(start_frame), video.time_at(end_frame)
    if end_frame <= start_frame:
        raise ValueError("flight end frame must follow its start frame")
    by_frame = index_ball_points(video, points)
    interval = [point for frame, point in sorted(by_frame.items()) if start_frame <= frame <= end_frame]
    observations = [point for point in interval if point.provenance in ("measured", "manual")]
    cutoff_px = max(3.0, video.width / 240)
    diagnostics = {
        "method": "gravity_only_v1", "start_frame": start_frame, "end_frame": end_frame,
        "gravity_m_s2": _GRAVITY.tolist(), "inlier_threshold_px": cutoff_px,
        "observed_frames": [point.frame for point in observations],
        "excluded_inferred_frames": [
            point.frame for point in interval if point.provenance in ("interpolated", "predicted")
        ],
    }

    def reject(reason):
        return FlightFit(None, {**diagnostics, "status": "unresolved", "reason": reason})

    if len(observations) < _MIN_POINTS:
        return reject("insufficient_measured_points")
    times = np.array([video.time_at(point.frame) for point in observations])
    span = float(times[-1] - times[0])
    diagnostics["observed_span_s"] = span
    if span < _MIN_SPAN_S:
        return reject("insufficient_time_span")
    origin_s = 0.5 * (times[0] + times[-1])
    pixels = np.array([(point.x_px, point.y_px) for point in observations])
    directions = np.array([camera.pixel_ray(pixel)[1] for pixel in pixels])

    # Each ray supplies two independent constraints perpendicular to its direction.
    projector = np.eye(3) - directions[:, :, None] * directions[:, None, :]
    elapsed = times - origin_s
    A = np.concatenate((projector, elapsed[:, None, None] / span * projector), axis=2)
    b = np.einsum("ijk,ik->ij", projector,
                  camera.center - 0.5 * elapsed[:, None] ** 2 * _GRAVITY)

    def residuals(parameters, indexes):
        world = _trajectory(parameters, times[indexes], origin_s, span)
        valid = camera.depth(world) > 1e-4
        errors = np.full((len(world), 2), float(video.width))
        if np.any(valid):
            errors[valid] = camera.project(world[valid]) - pixels[indexes][valid]
        return errors.ravel()

    def robust_support(errors):
        inliers = errors <= cutoff_px
        duration = np.ptp(times[inliers]) if inliers.any() else 0.0
        return inliers, duration >= max(_MIN_SPAN_S, 0.5 * span)

    # A bounded deterministic search provides seeds that survive occasional tracker outliers.
    all_indexes = np.arange(len(pixels))
    anchors = np.unique(np.rint(np.linspace(0, len(pixels) - 1, min(8, len(pixels)))).astype(int))
    seed = None
    best_score = (-1, -np.inf)
    for indexes in (all_indexes, *(np.array(triple) for triple in combinations(anchors, 3))):
        if times[indexes[-1]] - times[indexes[0]] < _MIN_SPAN_S:
            continue
        parameters, _, rank, singular = np.linalg.lstsq(A[indexes].reshape(-1, 6), b[indexes].ravel(), rcond=None)
        if rank != 6 or singular[0] > 1e5 * singular[-1]:
            continue
        errors = np.linalg.norm(residuals(parameters, all_indexes).reshape(-1, 2), axis=1)
        inliers, spans_arc = robust_support(errors)
        clipped_errors = np.minimum(errors, cutoff_px)
        score = (int(inliers.sum()), -float(np.sum(clipped_errors**2)))
        if spans_arc and score > best_score:
            seed, best_score = parameters, score
    required = max(_MIN_POINTS, int(np.ceil(0.60 * len(observations))))
    if seed is None or best_score[0] < required:
        return reject("no_consistent_ballistic_arc")

    inliers, _ = robust_support(np.linalg.norm(residuals(seed, all_indexes).reshape(-1, 2), axis=1))
    result = least_squares(
        residuals, seed, args=(np.flatnonzero(inliers),), loss="soft_l1",
        f_scale=cutoff_px / 2, x_scale="jac", max_nfev=100,
    )
    if not result.success:
        return reject("pixel_refinement_did_not_converge")
    singular = np.linalg.svd(result.jac, compute_uv=False)
    if singular[-1] <= 0 or singular[0] > 1e5 * singular[-1]:
        return reject("ill_conditioned_camera_rays")

    errors = np.linalg.norm(residuals(result.x, all_indexes).reshape(-1, 2), axis=1)
    inliers, spans_arc = robust_support(errors)
    diagnostics["inlier_frames"] = [point.frame for point, keep in zip(observations, inliers) if keep]
    diagnostics["reprojection_errors_px"] = {point.frame: float(error) for point, error in zip(observations, errors)}
    diagnostics["jacobian_condition"] = float(singular[0] / singular[-1])
    if inliers.sum() < required or not spans_arc:
        return reject("insufficient_inliers_after_refinement")

    # Report samples between supporting observations. Extension to unobserved contact or
    # bounce frames would make a different claim about the flight than this baseline can make.
    first, last = diagnostics["inlier_frames"][0], diagnostics["inlier_frames"][-1]
    flight_frames = range(first, last + 1)
    fitted_times = np.array([video.time_at(frame) for frame in flight_frames])
    xyz = _trajectory(result.x, fitted_times, origin_s, span)
    if np.any(camera.depth(xyz) <= 0):
        return reject("flight_has_nonpositive_camera_depth")
    if min(xyz[0, 2], xyz[-1, 2]) < 0:
        return reject("flight_crosses_court_surface")

    samples = tuple(MetricPoint(frame, *map(float, point)) for frame, point in zip(flight_frames, xyz))
    velocity = result.x[3:] / span + (fitted_times[0] - origin_s) * _GRAVITY
    diagnostics.update({
        "status": "accepted",
        "position_at_first_sample_m": xyz[0].tolist(),
        "velocity_at_first_sample_m_s": velocity.tolist(),
        "median_reprojection_px": float(np.median(errors[inliers])),
        "p90_reprojection_px": float(np.quantile(errors[inliers], 0.9)),
    })
    return FlightFit(MetricTrack("gravity_only_v1", samples), diagnostics)

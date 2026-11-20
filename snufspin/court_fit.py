"""Robust planar calibration and fixed-camera checks across video samples."""

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from .calibration import Calibration, Camera, CourtMapping, Intrinsics
from .court import LANDMARKS, LENGTH, DOUBLES_WIDTH
from .court_review import marking_support, markings_agree

CORNERS = np.array([[0, 0], [LENGTH, 0], [0, DOUBLES_WIDTH], [LENGTH, DOUBLES_WIDTH]])


@dataclass
class FrameFit:
    frame: int
    xyz: np.ndarray
    pixels: np.ndarray
    mapping: CourtMapping


def _physical_court(mapping):
    depths = CORNERS @ mapping.H[2, :2] + mapping.H[2, 2]
    return np.all(depths > 0) or np.all(depths < 0)


def _fit_mapping(xyz, pixels, intrinsics, threshold):
    if len(np.unique(xyz, axis=0)) < 4:
        return None
    ideal = pixels if intrinsics is None else intrinsics.undistort(pixels)
    H, mask = cv2.findHomography(xyz[:, :2], ideal, cv2.RANSAC, threshold)
    if H is None or not np.isfinite(H).all() or np.linalg.matrix_rank(H) != 3:
        return None
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 4 or inliers.mean() < 0.60:
        return None
    mapping = CourtMapping(H, intrinsics)
    if not _physical_court(mapping):
        return None
    return mapping, inliers


def estimate_intrinsics(mapping, width, height):
    """Orthogonality of the two court directions gives f² for centered K."""
    cx, cy = (width - 1) / 2, (height - 1) / 2
    directions = mapping.H[:, :2].copy()
    directions[:2] -= np.outer([cx, cy], directions[2])
    directions[:2] /= width
    directions /= np.linalg.norm(directions, axis=0)
    a, b = directions.T
    denominator = a[2] * b[2]
    if abs(denominator) < 1e-4:
        return None
    f_squared = -(a[:2] @ b[:2]) / denominator
    if not 0.25**2 <= f_squared <= 5.0**2:
        return None
    f = float(np.sqrt(f_squared) * width)
    return Intrinsics([[f, 0, cx], [0, f, cy], [0, 0, 1]])


def _solve_pose(xyz, pixels, intrinsics, threshold):
    xyz = np.ascontiguousarray(xyz)
    pixels = np.ascontiguousarray(pixels)
    K, dist = intrinsics.K, intrinsics.distortion
    _, rotations, translations, _ = cv2.solvePnPGeneric(xyz, pixels, K, dist, flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    for rvec, tvec in zip(rotations, translations):
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            continue
        rvec, tvec = cv2.solvePnPRefineLM(xyz, pixels, K, dist, rvec, tvec)
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            continue
        camera = Camera(intrinsics, cv2.Rodrigues(rvec)[0], tvec.ravel())
        corners = np.column_stack((CORNERS, np.zeros(len(CORNERS))))
        if camera.center[2] > 0 and np.all(camera.depth(corners) > 0):
            error = np.linalg.norm(camera.project(xyz) - pixels, axis=1)
            if np.quantile(error, 0.90) <= threshold:
                candidates.append((float(np.mean(error**2)), camera))
    return min(candidates, key=lambda pair: pair[0])[1] if candidates else None


def _mapping_disagreement(a, b):
    xy = np.unique(np.vstack((a.xyz, b.xyz)), axis=0)[:, :2]
    return np.quantile(np.linalg.norm(a.mapping.project(xy) - b.mapping.project(xy), axis=1), 0.90)


def _pose_spread(cameras):
    focal = [c.intrinsics.K[0, 0] for c in cameras]
    pairs = list(combinations(cameras, 2))
    return {
        "focal_relative_span": float(np.ptp(focal) / np.median(focal)),
        "center_span_m": float(max(np.linalg.norm(a.center - b.center) for a, b in pairs)),
        "rotation_span_deg": float(max(
            np.rad2deg(np.arccos(np.clip((np.trace(a.R @ b.R.T) - 1) / 2, -1, 1)))
            for a, b in pairs
        )),
    }


def fit_court(frames, landmarks, source, intrinsics=None):
    """Fit samples in source pixels and return (calibration or None, diagnostics)."""
    if not frames:
        raise ValueError("at least one decoded frame is required")
    height, width = next(iter(frames.values())).shape[:2]
    threshold = max(3.0, width / 240)
    fits = []
    diagnostics = {"frames": {}, "ransac_threshold_px": threshold}
    for frame, points in sorted(landmarks.items()):
        if frame not in frames:
            raise ValueError(f"landmarks refer to unsampled frame {frame}")
        names = list(points)
        xyz = np.array([LANDMARKS[name] for name in names], dtype=float).reshape(-1, 3)
        pixels = np.array([points[name] for name in names], dtype=float)
        if names and (pixels.shape != (len(names), 2) or not np.isfinite(pixels).all()
                      or np.any(pixels < 0) or np.any(pixels >= [width, height])):
            raise ValueError(f"landmarks for frame {frame} must be finite source pixels inside the frame")
        entry = {"observed": len(names)}
        diagnostics["frames"][str(frame)] = entry
        fitted = _fit_mapping(xyz, pixels, intrinsics, threshold)
        if fitted is None:
            continue
        mapping, mask = fitted
        support = marking_support(frames[frame], mapping)
        entry["inliers"] = [name for name, keep in zip(names, mask) if keep]
        entry["line_support"] = support
        if markings_agree(support):
            fits.append(FrameFit(frame, xyz[mask], pixels[mask], mapping))

    if not fits:
        diagnostics["status"] = "no_marking_supported_homography"
        return None, diagnostics
    disagreement = max((_mapping_disagreement(a, b) for a, b in combinations(fits, 2)), default=0)
    diagnostics["mapping_spread_p90_px"] = float(disagreement)
    if disagreement > 2 * threshold:
        diagnostics["status"] = "inconsistent_court_across_frames"
        return None, diagnostics

    xyz = np.concatenate([fit.xyz for fit in fits])
    pixels = np.concatenate([fit.pixels for fit in fits])
    pooled = _fit_mapping(xyz, pixels, intrinsics, threshold)
    if pooled is None:
        diagnostics["status"] = "no_shared_homography"
        return None, diagnostics
    mapping, mask = pooled
    supports = {str(index): marking_support(image, mapping) for index, image in frames.items()}
    diagnostics["shared_line_support"] = supports
    required = max(2, len(frames) // 2 + 1)
    if sum(map(markings_agree, supports.values())) < required:
        diagnostics["status"] = "shared_homography_failed_marking_check"
        return None, diagnostics

    candidates = []
    for fit in fits:
        frame_K = intrinsics or estimate_intrinsics(fit.mapping, width, height)
        pose = None if frame_K is None else _solve_pose(fit.xyz, fit.pixels, frame_K, threshold)
        entry = diagnostics["frames"][str(fit.frame)]
        entry["focal_px"] = None if frame_K is None else float(frame_K.K[0, 0])
        entry["camera_center_m"] = None if pose is None else pose.center.tolist()
        if pose is not None:
            candidates.append(pose)

    camera = None
    diagnostics["pose_status"] = "insufficient_independent_pose_estimates"
    if len(candidates) >= 2 and len(candidates) > len(fits) / 2:
        spread = _pose_spread(candidates)
        diagnostics["pose_spread"] = spread
        if (spread["focal_relative_span"] <= 0.15 and spread["center_span_m"] <= 1.0
                and spread["rotation_span_deg"] <= 2.0):
            pooled_K = intrinsics
            if pooled_K is None:
                K = candidates[0].intrinsics.K.copy()
                K[0, 0] = K[1, 1] = np.median([c.intrinsics.K[0, 0] for c in candidates])
                pooled_K = Intrinsics(K)
            camera = _solve_pose(xyz[mask], pixels[mask], pooled_K, threshold)
            if camera is not None:
                pose_support = {
                    str(index): marking_support(image, CourtMapping.from_camera(camera))
                    for index, image in frames.items()
                }
                diagnostics["pose_line_support"] = pose_support
                spread_with_pooled = _pose_spread([*candidates, camera])
                if (sum(map(markings_agree, pose_support.values())) < required
                        or spread_with_pooled["center_span_m"] > 1.0
                        or spread_with_pooled["rotation_span_deg"] > 2.0):
                    camera = None
            diagnostics["pose_status"] = "accepted" if camera is not None else "shared_pose_failed_checks"
        else:
            diagnostics["pose_status"] = "unstable_focal_length_or_pose"

    diagnostics["status"] = "camera" if camera is not None else "court_mapping_only"
    return Calibration(mapping, source, camera), diagnostics

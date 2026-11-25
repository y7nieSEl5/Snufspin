"""Calibrate a pinhole camera from matched court line segments."""

from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

from .calibration import Calibration, Camera, CourtMapping
from .court import LINES
from .court_fit import CORNERS, _physical_court
from .court_paint import score_mapping
from .line_court import _CROSS, _PAINT, _line_hypotheses, _prepare_segments, _segment_scores

VP_TOLERANCE = np.sin(np.deg2rad(4))
VP_PARTITIONS = 12
TEMPLATE_NAMES = tuple(LINES)
ACROSS = np.array([name in _CROSS for name in TEMPLATE_NAMES])
TEMPLATE_XYZ = np.concatenate((_PAINT, np.zeros((*_PAINT.shape[:2], 1))), axis=-1)
COURT_CORNERS = np.column_stack((CORNERS, np.zeros(len(CORNERS))))


def vp_partitions(segments, lengths, equations):
    """Propose two pencils of lines by angular agreement with homogeneous VPs."""
    if len(segments) < 4:
        return
    points = np.cross(*np.moveaxis(equations[np.array(list(combinations(range(len(segments)), 2)))], 1, 0))
    # Center and scale the projective coordinates before measuring angular errors.
    center = segments.mean(axis=(0, 1))
    scale = np.ptp(segments.reshape(-1, 2), axis=0).max()
    if scale <= 0:
        return
    points[:, :2] = (points[:, :2] - points[:, 2:] * center) / scale
    points = points[np.linalg.norm(points, axis=1) > 1e-9]
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    midpoints = (segments.mean(axis=1) - center) / scale
    vectors = points[:, None, :2] - points[:, None, 2:] * midpoints
    sine_error = np.abs(np.einsum("pnd,nd->pn", vectors, equations[:, :2]))
    magnitude = np.linalg.norm(vectors, axis=2)
    sine_error = np.divide(sine_error, magnitude, out=np.ones_like(sine_error), where=magnitude > 1e-9)

    partitions = {}
    indexes = np.arange(len(segments))
    for a, b in combinations(range(len(points)), 2):
        first = sine_error[a] <= sine_error[b]
        best_error = np.minimum(sine_error[a], sine_error[b])
        included = best_error <= VP_TOLERANCE
        groups = (tuple(indexes[first & included][:12]), tuple(indexes[~first & included][:12]))
        if min(map(len, groups)) < 2:
            continue
        # Require that each assigned pencil has support distinct from the other pencil.
        difference = np.abs(sine_error[a] - sine_error[b])
        if min(np.count_nonzero(difference[list(group)] >= VP_TOLERANCE / 2) for group in groups) < 2:
            continue
        score = float(np.sum(np.sqrt(lengths) * np.maximum(0, 1 - best_error / VP_TOLERANCE)))
        key = tuple(sorted(groups))
        previous = partitions.get(key)
        if previous is None or score > previous[0]:
            vps = points[[a, b]].copy()
            vps[:, :2] = vps[:, :2] * scale + vps[:, 2:] * center
            partitions[key] = score, groups, vps

    for _, groups, vps in sorted(partitions.values(), key=lambda x: -x[0])[:VP_PARTITIONS]:
        yield np.array(groups[0]), np.array(groups[1]), vps
        yield np.array(groups[1]), np.array(groups[0]), vps[::-1]


def direct_mapping(matches, equations, width, height):
    """Fit H from endpoint-on-supporting-line constraints in normalized coordinates."""
    model_ids = list(matches)
    across = sum(list(LINES)[i] in _CROSS for i in model_ids)
    if min(across, len(matches) - across) < 2:
        return None

    image_normalizer = np.array([[1 / width, 0, -0.5], [0, 1 / width, -height / (2 * width)], [0, 0, 1]])
    court_scale = np.ptp(_PAINT[:, :, 0])
    center = (_PAINT.max(axis=(0, 1)) + _PAINT.min(axis=(0, 1))) / 2
    court_normalizer = np.eye(3)
    court_normalizer[:2, :2] /= court_scale
    court_normalizer[:2, 2] = -center / court_scale
    image_to_pixels = np.linalg.inv(image_normalizer)
    lines = equations[list(matches.values())] @ image_to_pixels
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
    model = np.concatenate((_PAINT[model_ids], np.ones((len(matches), 2, 1))), axis=-1)
    model = model @ court_normalizer.T
    constraints = np.einsum("ni,nkj->nkij", lines, model).reshape(-1, 9)
    _, singular_values, basis = np.linalg.svd(constraints)
    if singular_values[7] < 1e-8 * singular_values[0]:
        return None
    H = image_to_pixels @ basis[-1].reshape(3, 3) @ court_normalizer
    if np.linalg.matrix_rank(H) != 3:
        return None
    mapping = CourtMapping(H)
    return mapping if _physical_court(mapping) else None


def pose_parameters(mapping, intrinsics):
    """Initialize a Euclidean pose from the direct segment homography."""
    normalized = np.linalg.solve(intrinsics.K, mapping.H)
    scale = 2 / (np.linalg.norm(normalized[:, 0]) + np.linalg.norm(normalized[:, 1]))
    scale *= np.sign(normalized[2, 2])
    a, b, t = (scale * normalized).T
    u, _, vh = np.linalg.svd(np.column_stack((a, b, np.cross(a, b))))
    rotation = u @ vh
    return np.r_[cv2.Rodrigues(rotation)[0].ravel(), t]


def segment_residuals(params, intrinsics, model, observed):
    rotation = cv2.Rodrigues(params[:3])[0]
    # Form the projected supporting line from homogeneous endpoints. No detected
    # intersections or correspondence between observed and template endpoints is needed.
    projected = (model @ rotation.T + params[3:]) @ intrinsics.K.T
    lines = np.cross(projected[:, 0], projected[:, 1])
    normals = np.maximum(np.linalg.norm(lines[:, :2], axis=1, keepdims=True), 1e-12)
    lines /= normals
    return (np.einsum("nei,ni->ne", observed, lines[:, :2]) + lines[:, 2:]).ravel()


def segment_support(mapping, prepared, width):
    """Independently rematch length-filtered segments to the scored pose's court."""
    segments, _, equations, _, _ = prepared
    if not len(segments):
        return {"score": 0.0, "along": 0, "across": 0}
    projected = mapping.project(_PAINT)
    scores = _segment_scores(projected[None], segments, max(4.0, width / 120))[0]
    # A lengthwise segment must point towards H[:,0], a transverse segment
    # towards H[:,1]. Homogeneous coordinates also handle a VP at infinity.
    vps = mapping.H[:, :2].T
    vectors = vps[:, None, :2] - vps[:, None, 2:] * segments.mean(axis=1)
    sine = abs(np.einsum("vsi,si->vs", vectors, equations[:, :2]))
    agrees = sine <= VP_TOLERANCE * np.linalg.norm(vectors, axis=-1)
    scores *= agrees[ACROSS.astype(int)]
    lines, observations = linear_sum_assignment(scores, maximize=True)
    weights = scores[lines, observations]
    supported = lines[weights >= .35]
    return {
        "score": float(weights.sum()),
        "along": int(np.count_nonzero(~ACROSS[supported])),
        "across": int(np.count_nonzero(ACROSS[supported])),
    }


def best_vp_hypothesis(prepared, width, height):
    """Match court markings using rectangle proposals from both VP families."""
    segments, lengths, equations, _, _ = prepared
    best = None
    for groups in vp_partitions(segments, lengths, equations):
        along, across, _ = groups
        candidates = _line_hypotheses(segments, equations, along, across, width, height, 1)
        if candidates and (best is None or candidates[0][0] > best[0][0]):
            best = candidates[0], groups
    return best


def fit_segment_pose(mapping, matched_segments, intrinsics):
    """Fit metric rotation and translation while keeping intrinsics fixed."""
    model = TEMPLATE_XYZ[[TEMPLATE_NAMES.index(name) for name in matched_segments]]
    observed = np.array(list(matched_segments.values()))
    start = pose_parameters(mapping, intrinsics)

    def objective(params):
        return segment_residuals(params, intrinsics, model, observed)

    fit = least_squares(objective, start, loss="soft_l1", f_scale=3.0, max_nfev=250)
    if not fit.success or np.linalg.matrix_rank(fit.jac) < 6:
        return None, {"status": "pose_fit_failed"}
    camera = Camera(intrinsics, cv2.Rodrigues(fit.x[:3])[0], fit.x[3:])
    if camera.center[2] <= 0 or np.any(camera.depth(COURT_CORNERS) <= 0):
        return None, {"status": "nonphysical_pose"}

    return camera, {
        "status": "fitted",
        "endpoint_line_p90_px": float(np.quantile(abs(objective(fit.x)), .90)),
    }


def score_segment_pose(mapping, prepared, gray, width):
    """Score the unchanged pose across the clip and test two-direction support."""
    segment_scores = {index: segment_support(mapping, lines, width) for index, lines in prepared.items()}
    paint = score_mapping(mapping, gray)
    supported = sum(
        paint.frames[index]["supported"] and min(row["along"], row["across"]) >= 2
        for index, row in segment_scores.items()
    )
    return {
        "segment_support": segment_scores,
        "median_segment_score": float(np.median([row["score"] for row in segment_scores.values()])),
        "paint_score": paint.score,
        "paint_support": paint.frames,
        "paint_frames": sum(row["supported"] for row in paint.frames.values()),
        "joint_support_frames": supported,
        "eligible": supported >= max(2, len(gray) // 2 + 1),
    }


def pose_rank(candidate):
    return not candidate["eligible"], -candidate["median_segment_score"], -candidate["paint_score"]


def fit_court_segments(frames, refined_segments, source, intrinsics):
    """Select a shared pose from fixed intrinsics and blind line-segment matches."""
    if len(frames) < 2 or frames.keys() != refined_segments.keys():
        raise ValueError("the segment solver needs matching segments on at least two sampled frames")
    if intrinsics.distortion is not None and np.any(intrinsics.distortion):
        raise ValueError("the segment solver needs pinhole intrinsics with zero distortion")
    height, width = next(iter(frames.values())).shape[:2]
    gray = {index: cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for index, frame in frames.items()}
    prepared = {
        index: _prepare_segments(segments, width, height)
        for index, segments in refined_segments.items()
    }
    diagnostics = {"frames": {}}
    ranked = []
    for index, lines in prepared.items():
        hypothesis = best_vp_hypothesis(lines, width, height)
        entry = diagnostics["frames"][str(index)] = {}
        if hypothesis is None:
            entry["status"] = "no_vanishing_direction_hypothesis"
            continue
        matches = hypothesis[0][2]
        matched_segments = {TEMPLATE_NAMES[model]: lines[0][segment] for model, segment in matches.items()}
        entry["matched_segments"] = {name: segment.tolist() for name, segment in matched_segments.items()}
        seed = direct_mapping(matches, lines[2], width, height)
        if seed is None:
            entry["status"] = "no_direct_segment_homography"
            continue

        camera, fit = fit_segment_pose(seed, matched_segments, intrinsics)
        entry.update(fit)
        if camera is None:
            continue
        mapping = CourtMapping.from_camera(camera)
        entry.update(score_segment_pose(mapping, prepared, gray, width))
        entry["camera_center_m"] = camera.center.tolist()
        ranked.append((index, camera, mapping, entry))

    if not ranked:
        diagnostics["status"] = "no_segment_pose"
        return None, diagnostics
    index, camera, mapping, top = min(ranked, key=lambda item: pose_rank(item[3]))
    diagnostics["top_frame"] = index
    if not top["eligible"]:
        diagnostics["status"] = "segment_pose_support_failed"
        return None, diagnostics
    diagnostics.update(status="camera", pose_status="accepted", winning_frame=index)
    return Calibration(mapping, {**source, "pose_method": "vp_segments"}, camera), diagnostics

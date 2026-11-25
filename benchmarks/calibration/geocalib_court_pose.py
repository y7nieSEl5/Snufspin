"""Fit metric CalTennis camera poses with fixed GeoCalib pinhole intrinsics."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

from snufspin.calibration import Camera, CourtMapping, Intrinsics
from snufspin.court import DOUBLES_WIDTH, LENGTH, LINES
from snufspin.court_commands import import_caltennis
from snufspin.court_paint import score_mapping
from snufspin.court_review import write_overlay
from snufspin.line_court import _CROSS, _PAINT, _prepare_segments, _segment_scores
from snufspin.video import Video

from .estimate import CLIPS, OUTPUT as BLIND
from .replay_segments import MANIFEST, VP_TOLERANCE
from .score import TRANSFORMS, pixel_grid, score_prediction
from .select_clip_fit import ANCHORS
from .transfer_fits import REPLAY, load_replay

OUTPUT = Path("outputs/caltennis-geocalib-pose")
TEMPLATE_NAMES = tuple(LINES)
ACROSS = np.array([name in _CROSS for name in TEMPLATE_NAMES])
TEMPLATE_XYZ = np.concatenate((_PAINT, np.zeros((*_PAINT.shape[:2], 1))), axis=-1)
CORNERS = np.array([[0, 0, 0], [LENGTH, 0, 0], [0, DOUBLES_WIDTH, 0], [LENGTH, DOUBLES_WIDTH, 0]])
MODE = "vp-segments"


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


def fit_pose(frame_fit, intrinsics):
    seed = frame_fit["modes"][MODE]
    if seed["H"] is None:
        return None, {"status": "no_direct_segment_homography"}

    segments = {item["id"]: item["refined"] for item in frame_fit["segments"]}
    matches = seed["matches"]
    model = TEMPLATE_XYZ[[TEMPLATE_NAMES.index(name) for name in matches]]
    observed = np.array([segments[index] for index in matches.values()])
    start = pose_parameters(CourtMapping(seed["H"]), intrinsics)

    def objective(params):
        return segment_residuals(params, intrinsics, model, observed)

    fit = least_squares(objective, start, loss="soft_l1", f_scale=3.0, max_nfev=250)
    if not fit.success or np.linalg.matrix_rank(fit.jac) < 6:
        return None, {"status": "pose_fit_failed", "matches": matches}
    camera = Camera(intrinsics, cv2.Rodrigues(fit.x[:3])[0], fit.x[3:])
    if camera.center[2] <= 0 or np.any(camera.depth(CORNERS) <= 0):
        return None, {"status": "nonphysical_pose", "matches": matches}

    return camera, {
        "status": "fitted",
        "matches": matches,
        "endpoint_line_p90_px": float(np.quantile(abs(objective(fit.x)), .90)),
    }


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


def estimate_clip(geo, frame_fits, frames):
    intrinsics = Intrinsics(np.median([row["K"] for row in geo["frames"].values()], axis=0))
    height, width = frames[0].shape[:2]
    gray = {sample["index"]: cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for sample, frame in zip(frame_fits, frames, strict=True)}
    prepared = {
        sample["index"]: _prepare_segments(
            np.array([item["refined"] for item in sample["segments"]]).reshape(-1, 2, 2), width, height,
        )
        for sample in frame_fits
    }
    required = len(frames) // 2 + 1
    entries = []
    for frame in frame_fits:
        camera, fit = fit_pose(frame, intrinsics)
        entry = {"source": frame["index"], "mode": MODE, **fit}
        if camera is not None:
            mapping = CourtMapping.from_camera(camera)
            segment_scores = {index: segment_support(mapping, lines, width) for index, lines in prepared.items()}
            paint = score_mapping(mapping, gray)
            paint_frames = sum(row["supported"] for row in paint.frames.values())
            joint_frames = sum(
                paint.frames[index]["supported"] and min(row["along"], row["across"]) >= 2
                for index, row in segment_scores.items()
            )
            entry.update({
                "camera": intrinsics.to_dict() | {"R": camera.R.tolist(), "t": camera.t.tolist()},
                "H": mapping.H.tolist(),
                "segment_support": segment_scores,
                "median_segment_score": float(np.median([row["score"] for row in segment_scores.values()])),
                "paint_score": paint.score,
                "paint_support": paint.frames,
                "paint_frames": paint_frames,
                "joint_support_frames": joint_frames,
                "eligible": joint_frames >= required,
            })
        entries.append(entry)

    ordered = sorted(
        (entry for entry in entries if "camera" in entry),
        key=lambda entry: (not entry["eligible"], -entry["median_segment_score"], -entry["paint_score"]),
    )
    # Retain the top estimate for diagnostics even if the gate declines it.
    top = ordered[0] if ordered else None
    return {
        "fixed_intrinsics": intrinsics.to_dict(),
        "top_frame": top["source"] if top else None,
        "status": "no_pose" if top is None else ("supported" if top["eligible"] else "support_gate_failed"),
        "candidates": entries,
    }


def score_camera(data, reference, pixels):
    camera = Camera(Intrinsics(data["K"], data["distortion"]), data["R"], data["t"])

    # Court paint is invariant to a 180-degree turn. Resolve only that naming
    # ambiguity for scoring, by comparing the planar template projection.
    flip = np.diag([-1., -1., 1.])
    origin = np.array([LENGTH, DOUBLES_WIDTH, 0.])
    opposite = Camera(camera.intrinsics, camera.R @ flip, camera.t + camera.R @ origin)
    points = np.column_stack((ANCHORS, np.zeros(len(ANCHORS))))
    expected = reference.project(points)
    candidates = (camera, opposite)
    errors = [float(np.quantile(np.linalg.norm(c.project(points) - expected, axis=1), .90))
              for c in candidates]
    label = int(np.argmin(errors))
    aligned = candidates[label]
    scored = data | {"R": aligned.R, "t": aligned.t}
    stats, _ = score_prediction(scored, reference, pixels)
    delta = aligned.center - reference.center
    rotation = aligned.R @ reference.R.T
    return stats | {
        "reference_flipped_180": bool(label),
        "template_p90_px": errors[label],
        "estimated_center_m": aligned.center.tolist(),
        "center_delta_xyz_m": delta.tolist(),
        "rotation_error_deg": float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(rotation)[0]))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=REPLAY)
    parser.add_argument("--geo", type=Path, default=BLIND / "geocalib.json")
    parser.add_argument("--references", type=Path, default=Path("data/caltennis/calib"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fits, frames = load_replay(args.replay)
    geo = json.loads(args.geo.read_text())
    manifest = json.loads(MANIFEST.read_text())
    estimates = {}
    for label, frame_fits in fits.items():
        info = geo["clips"][label]
        if info["samples"] != manifest[label] or set(info["frames"]) != {
            str(sample["index"]) for sample in manifest[label]
        }:
            raise ValueError(f"GeoCalib and the replay used different inputs for {label}")
        estimates[label] = estimate_clip(info, frame_fits, frames[label])
    (args.output / "blind.json").write_text(json.dumps(estimates, indent=2, allow_nan=False) + "\n")

    report = {}
    for label, estimate in estimates.items():
        stem = CLIPS[label]
        height, width = frames[label][0].shape[:2]
        video = Video(Path(stem + ".mp4"), width, height, ())
        pixels = pixel_grid(width, height)
        report[label] = {**estimate, "transforms": {}}
        for transform in TRANSFORMS:
            reference = import_caltennis(video, args.references / (stem + ".json"), transform).camera
            scores = {str(entry["source"]): score_camera(entry["camera"], reference, pixels)
                      for entry in estimate["candidates"] if "camera" in entry}
            report[label]["transforms"][transform] = {
                "supplied_center_m": reference.center.tolist(),
                "candidates": scores,
            }

        winner = next((entry for entry in estimate["candidates"] if entry["source"] == estimate["top_frame"]), None)
        if winner:
            mapping = CourtMapping(winner["H"])
            for fit, frame in zip(fits[label], frames[label], strict=True):
                write_overlay(args.output / f"{stem[:5]}-{fit['index']}.png", frame, mapping)

    (args.output / "scores.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

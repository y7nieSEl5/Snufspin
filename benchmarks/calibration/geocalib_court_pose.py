"""Fit metric CalTennis camera poses with fixed GeoCalib pinhole intrinsics."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from snufspin.calibration import Camera, CourtMapping, Intrinsics
from snufspin.court import DOUBLES_WIDTH, LENGTH
from snufspin.court_commands import import_caltennis
from snufspin.court_segments import fit_segment_pose, pose_rank, score_segment_pose
from snufspin.court_review import write_overlay
from snufspin.line_court import _prepare_segments
from snufspin.video import Video

from .estimate import CLIPS, OUTPUT as BLIND
from .replay_segments import MANIFEST
from .score import TRANSFORMS, pixel_grid, score_prediction
from .select_clip_fit import ANCHORS
from .transfer_fits import REPLAY, load_replay

OUTPUT = Path("outputs/caltennis-geocalib-pose")
MODE = "vp-segments"


def fit_pose(frame_fit, intrinsics):
    seed = frame_fit["modes"][MODE]
    if seed["H"] is None:
        return None, {"status": "no_direct_segment_homography"}

    segments = {item["id"]: item["refined"] for item in frame_fit["segments"]}
    matches = seed["matches"]
    matched = {name: segments[index] for name, index in matches.items()}
    camera, diagnostics = fit_segment_pose(CourtMapping(seed["H"]), matched, intrinsics)
    status = diagnostics.pop("status")
    return camera, {"status": status, "matches": matches, **diagnostics}


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
    entries = []
    for frame in frame_fits:
        camera, fit = fit_pose(frame, intrinsics)
        entry = {"source": frame["index"], "mode": MODE, **fit}
        if camera is not None:
            mapping = CourtMapping.from_camera(camera)
            entry.update({
                "camera": intrinsics.to_dict() | {"R": camera.R.tolist(), "t": camera.t.tolist()},
                "H": mapping.H.tolist(),
                **score_segment_pose(mapping, prepared, gray, width),
            })
        entries.append(entry)

    ordered = sorted(
        (entry for entry in entries if "camera" in entry),
        key=pose_rank,
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

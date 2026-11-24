"""Score previously saved estimates against the supplied CalTennis cameras."""

import argparse
import json
from pathlib import Path

import numpy as np

from snufspin.calibration import Camera, Intrinsics
from snufspin.court_commands import import_caltennis, read_json
from snufspin.video import Video

from .estimate import CLIPS, MANUAL, METHODS, OUTPUT

TRANSFORMS = ("resize", "center-padding")
GRID = (65, 37)


def pixel_grid(width, height):
    u, v = np.meshgrid(np.linspace(0, width - 1, GRID[0]), np.linspace(0, height - 1, GRID[1]))
    return np.column_stack((u.ravel(), v.ravel()))


def rays(intrinsics, pixels):
    xy = intrinsics.undistort(pixels, normalized=True)
    directions = np.column_stack((xy, np.ones(len(xy))))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def angles(a, b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a * b).sum(axis=-1)))


def angle_stats(values):
    median, p95 = np.quantile(values, [0.5, 0.95])
    return {"median": float(median), "p95": float(p95), "max": float(np.max(values))}


def score_prediction(prediction, reference, pixels):
    intrinsics = Intrinsics(prediction["K"], prediction.get("distortion"))
    estimated = rays(intrinsics, pixels)
    supplied = rays(reference.intrinsics, pixels)
    camera_angles = angles(estimated, supplied)
    focal = np.diag(intrinsics.K)[:2]
    result = {
        "focal_px": focal.tolist(),
        "focal_error_pct": (100 * (focal / np.diag(reference.intrinsics.K)[:2] - 1)).tolist(),
        "principal_point_px": intrinsics.K[:2, 2].tolist(),
        "camera_ray_deg": angle_stats(camera_angles),
    }
    if "R" in prediction:
        camera = Camera(intrinsics, prediction["R"], prediction["t"])
        result["court_ray_deg"] = angle_stats(angles(estimated @ camera.R, supplied @ reference.R))
        result["center_error_m"] = float(np.linalg.norm(camera.center - reference.center))
    return result, camera_angles


def score_view(entries, reference, size):
    pixels = pixel_grid(*size)
    methods = {}
    for method, entry in entries.items():
        result = {"status": entry["status"]}
        if "pose_status" in entry:
            result["pose_status"] = entry["pose_status"]
            result["observed_landmarks"] = {
                frame: info["observed"] for frame, info in entry["diagnostics"]["frames"].items()
            }
        predictions = entry.get("frames") or {}
        if method in ("geocalib", "anycalib") and set(predictions) != {
            str(sample["index"]) for sample in entry["samples"]
        }:
            raise ValueError(f"{method} must have one prediction for every sampled frame")
        if entry.get("camera") is not None:
            predictions = {"shared": entry["camera"]}

        if predictions:
            metrics = {frame: score_prediction(pred, reference, pixels)
                       for frame, pred in predictions.items()}
            rows = [row for row, _ in metrics.values()]
            result.update({
                "median_focal_px": np.median([row["focal_px"] for row in rows], axis=0).tolist(),
                "median_focal_error_pct": np.median([row["focal_error_pct"] for row in rows], axis=0).tolist(),
                "camera_ray_deg": angle_stats(np.concatenate([values for _, values in metrics.values()])),
                "predictions": {frame: row for frame, (row, _) in metrics.items()},
            })
            if "court_ray_deg" in rows[0]:
                result["court_ray_deg"] = rows[0]["court_ray_deg"]
                result["center_error_m"] = rows[0]["center_error_m"]
        methods[method] = result
    return methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimates", type=Path, default=OUTPUT)
    parser.add_argument("--references", type=Path, default=Path("data/caltennis/calib"))
    parser.add_argument("--output", type=Path, default=OUTPUT / "scores.json")
    parser.add_argument("--include-manual", action="store_true")
    args = parser.parse_args()

    methods = (*METHODS, MANUAL) if args.include_manual else METHODS
    estimates = {method: read_json(args.estimates / f"{method}.json") for method in methods}
    clips = {}
    for label, stem in CLIPS.items():
        entries = {method: report["clips"][label] for method, report in estimates.items()}
        anchor = entries["court"]
        for method, entry in entries.items():
            if any(entry[key] != anchor[key] for key in ("stem", "size", "samples")):
                raise ValueError(f"{method} used different input pixels for {label}")
        if anchor["stem"] != stem:
            raise ValueError(f"unexpected clip ID for {label}")
        video = Video(Path(f"{stem}.mp4"), *anchor["size"], ())
        clips[label] = {"size": anchor["size"], "samples": anchor["samples"], "transforms": {}}
        for transform in TRANSFORMS:
            calibration = import_caltennis(video, args.references / f"{stem}.json", transform)
            reference = calibration.camera
            clips[label]["transforms"][transform] = {
                "reference_intrinsics": reference.intrinsics.to_dict(),
                "calibration_sha256": calibration.source["calibration_sha256"],
                "methods": score_view(entries, reference, anchor["size"]),
            }

    report = {
        "ray_grid_width_height": GRID,
        "sources": {method: report["source"] for method, report in estimates.items()},
        "clips": clips,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

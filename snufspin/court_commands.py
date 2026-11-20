"""CLI entry points for importing and fitting court calibrations."""

from argparse import BooleanOptionalAction
import json
from pathlib import Path

import cv2
import numpy as np

from .calibration import Calibration, Camera, CourtMapping, Intrinsics, read_calibration
from .court import LANDMARKS
from .court_detector import TennisCourtDetector
from .court_fit import fit_court
from .court_review import write_overlay
from .tracking import file_sha256
from .video import read_sample_frames

DETECTOR = Path("data/tennis-court")


def add_court_commands(commands):
    fit = commands.add_parser("calibrate-court", help="fit a fixed camera or court mapping")
    fit.add_argument("video", type=Path)
    fit.add_argument("--landmarks", type=Path, help="JSON map of frame indices to named source pixels")
    fit.add_argument("--intrinsics", type=Path, help="JSON with decoded width, height, K and optional distortion")
    fit.add_argument("--detector", choices=("heatmap", "texflip"), default="texflip")
    fit.add_argument("--paint-fallback", action=BooleanOptionalAction, default=False,
                     help="try shared paint selection when TEXflip frame fits disagree on a fixed-camera clip")
    fit.add_argument("--model-source", type=Path, help="heatmap model Python file or TEXflip checkout")
    fit.add_argument("--weights", type=Path, help="checkpoint for the selected detector")
    fit.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    fit.add_argument("--threads", type=int, default=4)
    fit.add_argument("--samples", type=int, default=7, help="number of evenly spaced frame samples")
    fit.add_argument("--review-dir", type=Path, help="write overlays and diagnostics to this directory")

    import_cal = commands.add_parser("import-caltennis", help="convert a supplied CalTennis camera")
    import_cal.add_argument("video", type=Path)
    import_cal.add_argument("calibration", type=Path)
    import_cal.add_argument(
        "--pixel-transform", choices=("identity", "resize", "center-padding", "top-padding"),
        help="explicit calibration-to-decoded pixel transform when dimensions differ",
    )

    review = commands.add_parser("review-court", help="overlay a saved calibration on sampled frames")
    review.add_argument("video", type=Path)
    review.add_argument("calibration", type=Path)
    review.add_argument("--review-dir", type=Path, required=True)
    review.add_argument("--samples", type=int, default=7)
    review.add_argument("--frame", type=int, action="append", help="exact frame index, repeatable")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sample_indices(video, count):
    if count < 2:
        raise ValueError("request at least two samples to check the court across frames")
    frames = len(video.timestamps_s)
    return np.rint(np.linspace(0, frames - 1, min(count, frames))).astype(int).tolist()


def import_caltennis(video, path, transform):
    raw = read_json(path)
    if raw["camera_model"] != "pinhole":
        raise ValueError("expected a CalTennis pinhole camera")
    width, height = raw["image_shape"]["width"], raw["image_shape"]["height"]
    if Path(raw["annotation_file"]).stem != video.path.stem:
        raise ValueError("CalTennis annotation file refers to a different clip")
    if transform is None and (width, height) != (video.width, video.height):
        raise ValueError("CalTennis image size differs from decoded size; specify a checked --pixel-transform")
    A = np.eye(3)
    if transform == "resize":
        scale = np.array([video.width / width, video.height / height])
        A[:2, :2] = np.diag(scale)
        A[:2, 2] = (scale - 1) / 2
    elif transform in {"center-padding", "top-padding"}:
        if width != video.width or height > video.height:
            raise ValueError("expected vertical padding with matching image width")
        A[1, 2] = (video.height - height) // 2 if transform == "center-padding" else video.height - height
    camera = Camera(Intrinsics(A @ raw["K"], raw.get("dist_coeffs")), raw["R_w2c"], raw["t_w2c"])
    if camera.center[2] <= 0 or np.any(camera.depth(list(LANDMARKS.values())) <= 0):
        raise ValueError("CalTennis camera must be above the court with positive court depth")
    source = {
        "method": "CalTennis",
        "annotation_file": raw["annotation_file"],
        "calibration_sha256": file_sha256(path),
        "calibration_image_size": [width, height],
        "pixel_transform": transform or "identity",
        "calibration_to_decoded": A.tolist(),
        "published_reprojection_error_px": raw.get("reprojection_error"),
    }
    return Calibration(CourtMapping.from_camera(camera), source, camera)


def _try_paint_fallback(frames, segments, intrinsics, source, diagnostics):
    from .court_paint import select_paint_mapping

    if intrinsics is not None and intrinsics.distortion is not None and np.any(intrinsics.distortion):
        # Rank ideal straight lines in the undistorted grid. Keep the distortion
        # in the output mapping so callers can still use original video pixels.
        frames = {i: cv2.undistort(image, intrinsics.K, intrinsics.distortion)
                  for i, image in frames.items()}
        segments = {i: intrinsics.undistort(lines).reshape(lines.shape) if len(lines) else lines
                    for i, lines in segments.items()}
    winner, _, paint = select_paint_mapping(frames, segments)
    diagnostics["paint_fallback"] = {"trigger": diagnostics["status"], **paint}
    if winner is None:
        return None
    diagnostics["status"] = "court_mapping_only"
    diagnostics["pose_status"] = "paint_fallback_estimates_plane_only"
    return Calibration(CourtMapping(winner.mapping.H, intrinsics),
                       {**source, "mapping_method": "shared_paint_fallback"})


def run_court_command(video, args, parser):
    if args.command == "import-caltennis":
        report = import_caltennis(video, args.calibration, args.pixel_transform).to_report(video)
        print(json.dumps(report, indent=2, allow_nan=False))
        return

    if args.command == "review-court":
        calibration = read_calibration(video, read_json(args.calibration))
        indexes = args.frame if args.frame is not None else sample_indices(video, args.samples)
        args.review_dir.mkdir(parents=True, exist_ok=True)
        camera = calibration.camera if calibration.source["method"] != "CalTennis" else None
        for index, frame in read_sample_frames(video, indexes):
            write_overlay(args.review_dir / f"frame-{index:06}.png", frame, calibration.mapping, camera=camera)
        return

    intrinsics = None
    if args.intrinsics is not None:
        supplied = read_json(args.intrinsics)
        if (supplied["width"], supplied["height"]) != (video.width, video.height):
            raise ValueError("supplied intrinsics must already be expressed in decoded video pixels")
        intrinsics = Intrinsics(supplied["K"], supplied.get("distortion"))
    indexes = sample_indices(video, args.samples)
    segments = None
    if args.landmarks is None:
        frames = dict(read_sample_frames(video, indexes))
        if args.detector == "texflip":
            from .letr import LETR
            from .line_court import fit_lines, refine_segments

            detector = LETR(args.model_source or DETECTOR / "texflip",
                            args.weights or DETECTOR / "letr_best_checkpoint.pth",
                            args.device, args.threads)
            segments = {index: refine_segments(frame, detector(frame)) for index, frame in frames.items()}
            landmarks = {index: fit_lines(lines, video.width, video.height) for index, lines in segments.items()}
        else:
            detector = TennisCourtDetector(args.model_source or DETECTOR / "TennisCourtDetector/tracknet.py",
                                           args.weights or DETECTOR / "model.pth", args.device, args.threads)
            landmarks = {index: detector(frame) for index, frame in frames.items()}
        source = detector.source
    else:
        data = read_json(args.landmarks)
        landmarks = {int(index): points for index, points in data.items()}
        frames = dict(read_sample_frames(video, sorted(set(indexes).union(landmarks))))
        source = {"method": "manual_landmarks", "landmarks_sha256": file_sha256(args.landmarks)}

    source["intrinsics_method"] = "supplied" if intrinsics else "centered_vanishing_directions"
    if args.intrinsics is not None:
        source["intrinsics_sha256"] = file_sha256(args.intrinsics)
    calibration, diagnostics = fit_court(frames, landmarks, source, intrinsics)
    if (args.paint_fallback and segments is not None
            and calibration is None and diagnostics["status"] == "inconsistent_court_across_frames"):
        calibration = _try_paint_fallback(frames, segments, intrinsics, source, diagnostics)
    diagnostics["landmarks"] = landmarks
    if args.review_dir is not None:
        args.review_dir.mkdir(parents=True, exist_ok=True)
        (args.review_dir / "diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        if calibration is not None:
            for index, frame in frames.items():
                write_overlay(args.review_dir / f"frame-{index:06}.png", frame, calibration.mapping,
                              landmarks.get(index), calibration.camera)
    if calibration is None:
        parser.exit(1, f"snufspin: court calibration rejected ({diagnostics['status']})\n")
    report = calibration.to_report(video)
    report["diagnostics"] = diagnostics
    print(json.dumps(report, indent=2, allow_nan=False))

"""Estimate camera parameters from the CalTennis videos alone."""

import argparse
from bisect import bisect_left
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch

from snufspin.calibration import Intrinsics
from snufspin.court_commands import DETECTOR
from snufspin.court_fit import fit_court
from snufspin.tracking import file_sha256
from snufspin.video import Video, read_sample_frames

CLIPS = {
    "Mar 4 SW": "03_04_2026_17_42_22_000_1_SW_18_25_20_878",
    "Mar 9 SW": "03_09_2026_17_16_43_000_3_SW_02_17_39_13_016",
    "Oct 2 N": "10_02_2025_08_14_59_000_court5_N_2_08_30_22_535",
}
TIMES_S = (25, 300, 700)
METHODS = ("court", "geocalib", "anycalib")
MANUAL = "court-manual"
PAINT = Path(__file__).with_name("blind-paint-segments.json")
OUTPUT = Path("outputs/caltennis-blind")
MODELS = Path("data/calibration-models")


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def probe_sample_video(path):
    """Index the local CalTennis MP4s, which store one AVC frame per packet."""
    path = path.resolve(strict=True)
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "V:0",
        "-show_entries", "stream=width,height,codec_name,nb_frames:packet=pts_time",
        "-of", "json", str(path),
    ])
    data = json.loads(probe)
    stream, = data["streams"]
    timestamps = np.sort([float(p["pts_time"]) for p in data["packets"]])
    if (stream["codec_name"] != "h264" or len(timestamps) != int(stream["nb_frames"])
            or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0)):
        raise ValueError(f"expected one unique presentation timestamp per CalTennis frame in {path}")
    start = float(timestamps[0])
    return Video(path, stream["width"], stream["height"], tuple(timestamps - start), start)


def load_court_estimator(method, threads):
    from snufspin.line_court import fit_lines, refine_segments

    if method == MANUAL:
        from snufspin.court import LANDMARKS

        annotations = json.loads(PAINT.read_text())
        source = {"method": "blind manual paint segments", "seeds_sha256": file_sha256(PAINT)}

        def observations(label, image):
            lines = annotations[label]
            height, width = image.shape[:2]
            if (width, height) != (1920, 1088):
                raise ValueError("manual seeds require the original 1920 × 1088 decoded frames")
            seeds = 2 * np.asarray(list(lines.values()), dtype=float) + 0.5
            segments = refine_segments(image, seeds)
            a, b = np.moveaxis(np.dstack((segments, np.ones(segments.shape[:2]))), 1, 0)
            equations = dict(zip(lines, np.cross(a, b)))
            points = {}
            for name in LANDMARKS:
                kind, end, side = name.split("_", 2)
                transverse = f"{kind}_{end}"
                along = "center_service" if side == "center" else side
                if transverse in equations and along in equations:
                    x, y, z = np.cross(equations[transverse], equations[along])
                    if z != 0 and 0 <= x / z < width and 0 <= y / z < height:
                        points[name] = [x / z, y / z]
            return points
    else:
        from snufspin.letr import LETR

        upstream = DETECTOR / "texflip"
        detector = LETR(upstream, DETECTOR / "letr_best_checkpoint.pth", "cpu", threads)
        source = {**detector.source, "revision": revision(upstream),
                  "letr_revision": revision(upstream / "LETR")}

        def observations(label, image):
            height, width = image.shape[:2]
            return fit_lines(refine_segments(image, detector(image)), width, height)

    def estimate(label, frames):
        landmarks = {index: observations(label, image) for index, image in frames.items()}
        fit, diagnostics = fit_court(frames, landmarks, source)
        camera = fit.camera if fit else None
        return {
            "status": diagnostics["status"],
            "pose_status": diagnostics.get("pose_status"),
            "camera": None if camera is None else {
                **camera.intrinsics.to_dict(), "R": camera.R.tolist(), "t": camera.t.tolist(),
            },
            "diagnostics": diagnostics,
        }

    return estimate, source


def load_estimator(method, models, threads):
    if method in ("court", MANUAL):
        return load_court_estimator(method, threads)

    source = models / ("GeoCalib" if method == "geocalib" else "AnyCalib")
    sys.path.insert(0, str(source.resolve()))
    torch.hub.set_dir(str(models / "hub"))
    if method == "geocalib":
        from geocalib import GeoCalib

        weights = models / "hub/geocalib/pinhole.tar"
        model = GeoCalib(weights=str(weights)).eval()

        def predict(image):
            result = model.calibrate(image, camera_model="pinhole")
            return result["camera"].K[0].cpu().numpy()
    else:
        from anycalib import AnyCalib

        weights = models / "hub/anycalib/anycalib_pinhole.pt"
        if not weights.is_file():
            raise FileNotFoundError(weights)
        model = AnyCalib(model_id="anycalib_pinhole").eval()

        def predict(image):
            result = model.predict(image, cam_id="pinhole")
            fx, fy, cx, cy = result["intrinsics"].cpu().numpy()
            # AnyCalib uses (0.5, 0.5) as the first pixel center.
            return [[fx, 0, cx - 0.5], [0, fy, cy - 0.5], [0, 0, 1]]

    def estimate(label, frames):
        predictions = {}
        for index, bgr in frames.items():
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
            predictions[str(index)] = Intrinsics(predict(image)).to_dict()
        return {"status": "intrinsics", "frames": predictions}

    return estimate, {"revision": revision(source), "weights_sha256": file_sha256(weights)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=(*METHODS, MANUAL))
    parser.add_argument("--videos", type=Path, default=Path("data/caltennis/originals"))
    parser.add_argument("--models", type=Path, default=MODELS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    cv2.setRNGSeed(0)
    estimate, source = load_estimator(args.method, args.models, args.threads)
    clips = {}
    for label, stem in CLIPS.items():
        print(f"{args.method}: {label}", flush=True)
        video = probe_sample_video(args.videos / f"{stem}.mp4")
        indexes = [bisect_left(video.timestamps_s, t) for t in TIMES_S]
        frames = dict(read_sample_frames(video, indexes))
        clips[label] = {
            "stem": stem,
            "size": [video.width, video.height],
            "samples": [
                {"index": index, "time_s": video.time_at(index),
                 "bgr_sha256": hashlib.sha256(frame).hexdigest()}
                for index, frame in frames.items()
            ],
            **estimate(label, frames),
        }
        print(f'  {clips[label]["status"]}', flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.method}.json"
    path.write_text(json.dumps({"source": source, "clips": clips}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

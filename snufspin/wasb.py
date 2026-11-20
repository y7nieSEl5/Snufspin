"""Adapt the official WASB Tennis detector to Snufspin's video and track format."""

from collections import defaultdict, deque
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys

from .tracking import BallPoint, file_sha256, track_report
from .video import Video, decode_bgr_frames


REVISION = "923462cacdeb3353b84ddebdedb3f4b7a8553b0f"
WEIGHTS_SHA256 = "9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb"


@dataclass
class WasbCandidateLog:
    first_frame: int
    last_frame: int
    frames: list[dict] = field(default_factory=list, init=False)

    def select(self, tracker, frame_index, candidates):
        if not self.first_frame <= frame_index <= self.last_frame:
            return tracker.update(candidates)

        # Consult the pinned upstream gate before update advances its history.
        passed = {id(candidate) for candidate in tracker._select_not_too_far(candidates)}
        entries = [
            {
                "x_px": float(candidate["xy"][0]),
                "y_px": float(candidate["xy"][1]),
                "score": float(candidate["score"]),
                "passed_gate": id(candidate) in passed,
            }
            for candidate in candidates
        ]
        selected = tracker.update(candidates)
        winner = None
        if selected["visi"]:
            winner = next(
                index for index, entry in enumerate(entries)
                if entry["passed_gate"]
                and (entry["x_px"], entry["y_px"], entry["score"])
                == (selected["x"], selected["y"], selected["score"])
            )
        self.frames.append({"frame": frame_index, "candidates": entries, "selected_index": winner})
        return selected


def _selected_frames(frames, window_size, detect, tracker, stride=1, candidate_log=None):
    """Select each frame after all sampled windows that include it have run."""
    window = deque(maxlen=window_size)
    pending = defaultdict(list)
    next_frame = 0
    frame_count = 0

    def select(frame_index):
        candidates = pending.pop(frame_index, [])
        if candidate_log is None:
            return tracker.update(candidates)
        return candidate_log.select(tracker, frame_index, candidates)

    for frame in frames:
        window.append(frame)
        frame_count += 1
        start = frame_count - window_size
        if start < 0 or start % stride != 0:
            continue
        # The upstream Tennis loader assigns the oldest output to the first
        # frame of the window when frames_in == frames_out.
        for offset, candidates in enumerate(detect(window)):
            pending[start + offset].extend(candidates)
        for frame_index in range(start, start + stride):
            yield frame_index, select(frame_index)
        next_frame = start + stride
    for frame_index in range(next_frame, frame_count):
        yield frame_index, select(frame_index)


def run_wasb(
    video: Video, upstream: Path, weights: Path, device: str = "cpu", threads: int = 4, stride: int = 1,
    *, candidate_log: WasbCandidateLog | None = None,
) -> dict:
    if threads < 1:
        raise ValueError("--threads must be positive")
    if stride not in (1, 3):
        raise ValueError("--stride must be 1 or 3")
    if candidate_log is not None and not (
        0 <= candidate_log.first_frame <= candidate_log.last_frame < len(video.timestamps_s)
    ):
        raise ValueError("--candidate-frames must be an inclusive range within the video")
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise ValueError("--upstream must point to the WASB-SBDT git checkout") from error
    if revision != REVISION or not (upstream / "src" / "configs").is_dir():
        raise ValueError(f"expected a WASB-SBDT checkout at {REVISION}")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--", "src"]):
        raise ValueError("the pinned WASB-SBDT src directory must be clean")
    if not weights.is_file() or file_sha256(weights) != WEIGHTS_SHA256:
        raise ValueError("--weights must be the official WASB (Ours) Tennis checkpoint; see README.md")

    # WASB uses top-level imports rooted at its src directory.
    sys.path.insert(0, str((upstream / "src").resolve()))
    try:
        import cv2
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from PIL import Image
        from dataloaders import build_img_transforms, get_transform
        from detectors.postprocessor import TracknetV2Postprocessor
        from models import build_model
        from trackers import build_tracker
    except ImportError as error:
        raise ValueError("install WASB's dependencies with pip install -r requirements-wasb.txt") from error
    finally:
        sys.path.pop(0)

    config_dir = upstream / "src" / "configs"
    cfg = {
        "model": OmegaConf.load(config_dir / "model" / "wasb.yaml"),
        "detector": OmegaConf.load(config_dir / "detector" / "tracknetv2.yaml"),
        "dataloader": OmegaConf.load(config_dir / "dataloader" / "default.yaml"),
        "tracker": OmegaConf.load(config_dir / "tracker" / "online.yaml"),
    }
    post_cfg = cfg["detector"]["postprocessor"]
    input_wh = (cfg["model"]["inp_width"], cfg["model"]["inp_height"])
    output_wh = (cfg["model"]["out_width"], cfg["model"]["out_height"])
    window_size = cfg["model"]["frames_in"]
    _, transform = build_img_transforms(cfg)

    torch.set_num_threads(threads)
    model = build_model(cfg)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    postprocessor = TracknetV2Postprocessor(cfg)
    tracker = build_tracker(cfg)

    # The affine transforms depend only on the source frame dimensions.
    frame_shape = (video.height, video.width, 3)
    geometry = np.empty(frame_shape, dtype=np.uint8)
    forward = get_transform(geometry, input_wh)
    inverse = {0: torch.from_numpy(get_transform(geometry, output_wh, inv=1)).unsqueeze(0)}

    def prepare(pixels):
        bgr = np.frombuffer(pixels, dtype=np.uint8).reshape(frame_shape)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        warped = cv2.warpAffine(rgb, forward, input_wh, flags=cv2.INTER_LINEAR)
        return transform(Image.fromarray(warped))

    def detect(window):
        tensor = torch.cat(tuple(window), dim=0).unsqueeze(0).to(device)
        outputs = postprocessor.run(model(tensor), inverse)[0]
        for offset in range(cfg["model"]["frames_out"]):
            heatmap = outputs[offset][0]
            yield [
                {"xy": xy, "score": score, "scale": 0}
                for xy, score in zip(heatmap["xys"], heatmap["scores"])
            ]

    points = []
    # The upstream tracker spells infinity np.Inf, removed in NumPy 2.
    needs_inf_alias = not hasattr(np, "Inf")
    if needs_inf_alias:
        np.Inf = np.inf
    try:
        with torch.inference_mode(), closing(decode_bgr_frames(video)) as frames:
            for frame_index, result in _selected_frames(
                map(prepare, frames), window_size, detect, tracker, stride, candidate_log,
            ):
                x, y = float(result["x"]), float(result["y"])
                if result["visi"] and 0 <= x < video.width and 0 <= y < video.height:
                    points.append(BallPoint(frame_index, x, y, "measured"))
    finally:
        if needs_inf_alias:
            del np.Inf

    return track_report(video, tuple(points), {
        "engine": "wasb-tennis",
        "upstream_revision": revision,
        "weights_sha256": WEIGHTS_SHA256,
        "input_width": input_wh[0],
        "input_height": input_wh[1],
        "stride": stride,
        "frames_in": window_size,
        "frames_out": cfg["model"]["frames_out"],
        "batch_size": 1,
        "device": device,
        "threads": threads,
        "heatmap_threshold": post_cfg["score_threshold"],
        "heatmap_scale": post_cfg["scales"][0],
        "blob_detection": post_cfg["blob_det_method"],
        "use_heatmap_weight": post_cfg["use_hm_weight"],
        "tracker": cfg["tracker"]["name"],
        "max_displacement_px": cfg["tracker"]["max_disp"],
    })

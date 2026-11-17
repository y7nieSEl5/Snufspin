"""Local ball detection adapted from tennis-vision's pinned detector.

The BallTrackerNet architecture is loaded from the external TennisProject
checkout. Project attribution and the upstream MIT notice are in CREDITS.md.
"""

from collections import deque
from contextlib import closing
from importlib.util import module_from_spec, spec_from_file_location
from math import dist
from pathlib import Path

from .tracking import BallPoint, file_sha256, track_report
from .video import Video, decode_bgr_frames


INPUT_WIDTH = 640
INPUT_HEIGHT = 360


def select_ball(heatmap, previous, scale_x, scale_y):
    """Preserve the reviewed baseline's mask conversion and candidate gate."""
    import cv2
    import numpy as np

    mask_values = (heatmap * 255).astype(np.uint8)
    _, mask = cv2.threshold(mask_values, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        mask, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
        param1=50, param2=2, minRadius=2, maxRadius=7,
    )
    if circles is None:
        return None
    maximum_step = 80.0 * max(scale_x, scale_y) / 2.0
    for x, y, _ in circles[0]:
        candidate = (float(x) * scale_x, float(y) * scale_y)
        if previous is None or not previous[0] or dist(candidate, previous) < maximum_step:
            return candidate
    return None


def run_tracknet(
    video: Video, model_source: Path, weights: Path, device: str = "cpu", threads: int = 4,
) -> dict:
    if not model_source.is_file() or not weights.is_file():
        raise ValueError("TrackNet requires TennisProject's tracknet.py and a checkpoint. See README.md.")
    if threads < 1:
        raise ValueError("--threads must be positive")

    try:
        import cv2
        import numpy as np
        import torch
    except ImportError as error:
        raise ValueError("install the tracking dependencies with pip install -r requirements.txt") from error

    torch.set_num_threads(threads)

    # The model module and checkpoint are explicitly supplied by the caller.
    spec = spec_from_file_location("snufspin_upstream_tracknet", model_source)
    if spec is None or spec.loader is None:
        raise ValueError("--model-source must point to TennisProject's tracknet.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.BallTrackerNet(input_channels=9, out_channels=256)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    points = []
    previous = None
    window = deque(maxlen=3)
    with torch.inference_mode(), closing(decode_bgr_frames(video)) as frames:
        for frame_index, pixels in enumerate(frames):
            frame = np.frombuffer(pixels, dtype=np.uint8).reshape(video.height, video.width, 3)
            window.appendleft(cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT)))
            if len(window) < 3:
                continue
            # The checkpoint expects BGR channels, newest frame first.
            stack = np.concatenate(tuple(window), axis=2).transpose(2, 0, 1)
            tensor = torch.from_numpy(stack).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
            heatmap = model(tensor).argmax(dim=1).reshape(INPUT_HEIGHT, INPUT_WIDTH).cpu().numpy()
            previous = select_ball(heatmap, previous, video.width / INPUT_WIDTH, video.height / INPUT_HEIGHT)
            if previous is None:
                continue
            points.append(BallPoint(
                frame_index, previous[0], previous[1], "measured",
            ))

    return track_report(video, tuple(points), {
        "engine": "tennis-vision-compatible-local",
        "model_sha256": file_sha256(model_source),
        "weights_sha256": file_sha256(weights),
        "device": device,
        "threads": threads,
    })

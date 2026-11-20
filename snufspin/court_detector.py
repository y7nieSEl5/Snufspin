"""Run the published TennisCourtDetector heatmap model on decoded frames."""

import importlib.util
from pathlib import Path

import cv2
import numpy as np

from .tracking import file_sha256

# Upstream channels follow the broadcast convention, far baseline first.
CHANNELS = (
    "baseline_b_doubles_high", "baseline_b_doubles_low",
    "baseline_a_doubles_high", "baseline_a_doubles_low",
    "baseline_b_singles_high", "baseline_a_singles_high",
    "baseline_b_singles_low", "baseline_a_singles_low",
    "service_b_singles_high", "service_b_singles_low",
    "service_a_singles_high", "service_a_singles_low",
    "service_b_center", "service_a_center",
)
MODEL_WIDTH = 640
MODEL_HEIGHT = 360


class TennisCourtDetector:
    def __init__(self, model_source: Path, weights: Path, device: str, threads: int):
        import torch

        if threads < 1:
            raise ValueError("threads must be positive")
        spec = importlib.util.spec_from_file_location(
            "_snufspin_tennis_court_detector", model_source.resolve(strict=True),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        torch.set_num_threads(threads)
        self.device = torch.device(device)
        self.model = module.BallTrackerNet(out_channels=15)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model.to(self.device).eval()
        self.source = {
            "method": "TennisCourtDetector",
            "model_sha256": file_sha256(model_source),
            "weights_sha256": file_sha256(weights),
        }

    def __call__(self, image):
        import torch

        height, width = image.shape[:2]
        resized = cv2.resize(image, (MODEL_WIDTH, MODEL_HEIGHT))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
        with torch.inference_mode():
            heatmaps = self.model(tensor.to(self.device, dtype=torch.float32) / 255)[0].sigmoid()
        points = {}
        for name, heatmap in zip(CHANNELS, heatmaps.cpu().numpy()):
            # Take the centroid of the confident component containing the peak.
            # Keep it independent of any homography so RANSAC sees the raw evidence.
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            if heatmap[y, x] < 0.67:
                continue
            _, labels, stats, centers = cv2.connectedComponentsWithStats(
                (heatmap >= 0.67).astype(np.uint8),
            )
            component = labels[y, x]
            if stats[component, cv2.CC_STAT_AREA] < 4:
                continue
            u, v = centers[component]
            points[name] = (
                float(np.clip((u + 0.5) * width / MODEL_WIDTH - 0.5, 0, width - 1)),
                float(np.clip((v + 0.5) * height / MODEL_HEIGHT - 0.5, 0, height - 1)),
            )
        return points

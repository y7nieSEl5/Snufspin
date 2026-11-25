"""Optional GeoCalib pinhole estimates for fixed-camera video samples."""

from pathlib import Path

import cv2
import numpy as np

from .calibration import Intrinsics
from .tracking import file_sha256


def estimate_intrinsics(frames, device, weights=None):
    """Return one pinhole matrix in decoded pixels and its model provenance."""
    import torch

    try:
        from geocalib import GeoCalib
    except ModuleNotFoundError as error:
        if error.name != "geocalib":
            raise
        raise ValueError("install GeoCalib to estimate intrinsics, or pass --intrinsics") from error

    matrices = []
    # GeoCalib initializes its matrix decomposition from the CPU generator,
    # including when the image is on an accelerator.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.random.default_generator.manual_seed(0)
        model = GeoCalib(weights=str(weights) if weights is not None else "pinhole").to(device).eval()
        for bgr in frames.values():
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(rgb).permute(2, 0, 1).to(device=device, dtype=torch.float32) / 255
            result = model.calibrate(image, camera_model="pinhole")
            matrices.append(result["camera"].K[0].cpu().numpy())

    checkpoint = weights if weights is not None else Path(torch.hub.get_dir()) / "geocalib/pinhole.tar"
    return Intrinsics(np.median(matrices, axis=0)), {
        "intrinsics_method": "geocalib_pinhole_median",
        "intrinsics_weights_sha256": file_sha256(checkpoint),
    }

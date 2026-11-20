"""Inference adapter for TEXflip's tennis-trained LETR checkpoint."""

from argparse import Namespace
import importlib
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from .tracking import file_sha256


class LETR:
    def __init__(self, upstream: Path, weights: Path, device: str, threads: int):
        import torch
        import torch.nn.modules.linear as linear
        import torchvision

        if threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
        source = (upstream / "LETR/src").resolve(strict=True)
        with torch.serialization.safe_globals([Namespace]):
            checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        args = checkpoint["args"]
        args.device = "cpu"

        class AttentionWithoutWeights(torch.nn.MultiheadAttention):
            def forward(self, query, key, value, **kwargs):
                # The second stage consumes only the output, so use PyTorch's SDPA path.
                return super().forward(query, key, value, **{**kwargs, "need_weights": False})

        # LETR's legacy version check interprets TorchVision 0.29 as 0.2.
        with (
            patch.object(sys, "path", [str(source), *sys.path]),
            patch.object(torchvision, "__version__", "1.0"),
            patch.object(linear, "_LinearWithBias", getattr(linear, "_LinearWithBias", torch.nn.Linear),
                         create=True),
        ):
            model_module = importlib.import_module("models.letr")
            backbone_module = importlib.import_module("models.backbone")
            with patch.object(backbone_module, "is_main_process", return_value=False):
                backbone = model_module.build_backbone(args)
            model = model_module.LETR(
                backbone, model_module.build_transformer(args), num_classes=1,
                num_queries=args.num_queries, args=args, aux_loss=args.aux_loss,
            )
            if args.LETRpost:
                with patch.object(torch.nn, "MultiheadAttention", AttentionWithoutWeights):
                    model = model_module.LETRstack(model, args=args)
        model.load_state_dict(checkpoint["model"])
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.source = {
            "method": "TEXflip LETR with line-based court fitting",
            "weights_sha256": file_sha256(weights),
        }

    def __call__(self, image):
        import torch
        from torchvision.transforms import functional

        height, width = image.shape[:2]
        # Preserve the BGR order and normalization of TEXflip's inference script.
        tensor = functional.to_tensor(image.copy())
        tensor = functional.normalize(tensor, [0.538, 0.494, 0.453], [0.257, 0.263, 0.273])
        tensor = functional.resize(tensor, [1100], antialias=False)
        with torch.inference_mode():
            prediction = self.model([tensor.to(self.device)])
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            scores = prediction["pred_logits"][0].softmax(-1)[:, 0]
            lines = prediction["pred_lines"][0][scores >= 0.5].cpu().numpy()
        return lines.reshape(-1, 2, 2) * np.array([width, height])

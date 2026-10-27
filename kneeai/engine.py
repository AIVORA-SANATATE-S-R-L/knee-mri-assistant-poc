"""Safe loading and execution of the third-party knee MRI model.

Differences from the upstream KneeMRIInferenceEngine:
  * the checkpoint SHA-256 is verified against the pinned value before loading;
  * torch.load is called with weights_only=True (no arbitrary code execution
    from the pickle); if that fails we STOP instead of falling back;
  * state-dict keys are loaded strictly and the result is recorded;
  * outputs are checked to be finite.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR_PARENT = os.path.join(ROOT, "third_party")
VENDOR_DIR = os.path.join(VENDOR_PARENT, "rsna_knee_kaggle_model")

PINNED_COMMIT = "a254152c1e7cf4aa1d7391cb9260d54a4185118c"
PINNED_SHA256 = "68bd5d3fd040fa6e4c464d99d2c7dcacaebd2a40388ec0902133d4e26be0f791"
PINNED_SIZE = 152040442

TARGETS_REPORTED = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"]


def _vendor():
    if VENDOR_PARENT not in sys.path:
        sys.path.insert(0, VENDOR_PARENT)
    from rsna_knee_kaggle_model.model import HierarchicalKneeMRIModel  # type: ignore
    from rsna_knee_kaggle_model.preprocessing import build_study_input_tensor  # type: ignore
    return HierarchicalKneeMRIModel, build_study_input_tensor


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelCheckError(RuntimeError):
    pass


class SafeKneeEngine:
    def __init__(self, model_dir: str = VENDOR_DIR, device: Optional[str] = None, verify_hash: bool = True):
        self.model_dir = model_dir
        ckpt = os.path.join(model_dir, "best_model.pth")
        if not os.path.exists(ckpt):
            raise ModelCheckError(f"Checkpoint not found: {ckpt}. Run scripts/setup_windows.ps1 first.")
        size = os.path.getsize(ckpt)
        if size < 1_000_000:
            raise ModelCheckError(f"Checkpoint is only {size} bytes: this is a Git LFS pointer, not the weights.")
        self.ckpt_size = size
        self.ckpt_sha256 = sha256_of(ckpt)
        if verify_hash and self.ckpt_sha256 != PINNED_SHA256:
            raise ModelCheckError(
                f"Checkpoint hash {self.ckpt_sha256} does not match pinned {PINNED_SHA256}. "
                "Do not substitute another checkpoint silently.")

        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
            self.config = json.load(fh)
        with open(os.path.join(model_dir, "thresholds.json"), encoding="utf-8") as fh:
            self.thresholds = json.load(fh)
        self.targets = self.config["target_columns"]
        self.data_cfg = self.config.get("data", {})
        mcfg = self.config.get("model", {})

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        Model, self._build = _vendor()
        self.model = Model(
            encoder_name=mcfg.get("encoder_name", "resnet18"),
            feature_dim=mcfg.get("feature_dim", 512),
            slice_pooling=mcfg.get("slice_pooling", "mean_max"),
            series_pooling=mcfg.get("series_pooling", "mean_max"),
            use_metadata=mcfg.get("use_metadata", True),
            metadata_dim=mcfg.get("metadata_dim", 64),
            hidden_dim=mcfg.get("hidden_dim", 256),
            num_classes=len(self.targets),
            dropout=0.0,
        )
        try:
            checkpoint = torch.load(ckpt, map_location=self.device, weights_only=True)
        except Exception as exc:
            raise ModelCheckError(
                "Safe loading (weights_only=True) failed. STOP: request a plain state_dict or "
                f"safetensors export from the project lead. Do not use weights_only=False. Detail: {exc}")
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        result = self.model.load_state_dict(state, strict=True)
        self.load_report = {
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "checkpoint_top_level_keys": list(checkpoint.keys())[:20] if isinstance(checkpoint, dict) else [],
        }
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def _forward(self, images, metadata, mask) -> np.ndarray:
        logits = self.model(images.to(self.device), metadata=metadata.to(self.device), series_mask=mask.to(self.device))
        if not torch.isfinite(logits).all():
            raise RuntimeError("Model produced non-finite outputs")
        return torch.sigmoid(logits).cpu().numpy()[0]

    def dummy_check(self) -> Dict[str, Any]:
        """Run on zero/random tensors: proves only that the software executes."""
        n = self.data_cfg.get("num_slices_per_series", 16)
        s = self.data_cfg.get("max_series_per_study", 6)
        size = self.data_cfg.get("image_size", 224)
        g = torch.Generator().manual_seed(0)
        images = torch.rand((1, s, n, 1, size, size), generator=g)
        meta = torch.zeros((1, s, 10)); meta[..., 2] = 1; meta[..., 5] = 1; meta[..., 9] = 1
        mask = torch.ones((1, s))
        t0 = time.time()
        probs = self._forward(images, meta, mask)
        return {"finite": bool(np.isfinite(probs).all()), "seconds": round(time.time() - t0, 3),
                "probabilities": {t: float(p) for t, p in zip(self.targets, probs)}}

    def predict_staged(self, staged_dir: str, metadata_df: pd.DataFrame) -> Dict[str, Any]:
        inputs = self._build(
            study_dir=staged_dir,
            series_metadata_df=metadata_df,
            image_size=self.data_cfg.get("image_size", 224),
            num_slices=self.data_cfg.get("num_slices_per_series", 16),
            max_series=self.data_cfg.get("max_series_per_study", 6),
            intensity_norm=self.data_cfg.get("intensity_norm", "percentile"),
            p_min=self.data_cfg.get("p_min", 1.0),
            p_max=self.data_cfg.get("p_max", 99.0),
        )
        t0 = time.time()
        probs = self._forward(inputs["images"], inputs["metadata"], inputs["series_mask"])
        elapsed = time.time() - t0
        return {
            "probabilities": {t: float(p) for t, p in zip(self.targets, probs)},
            "thresholds": {t: float(self.thresholds.get(t, 0.5)) for t in self.targets},
            "binary": {t: int(p >= self.thresholds.get(t, 0.5)) for t, p in zip(self.targets, probs)},
            "inference_seconds": round(elapsed, 3),
            "images_tensor": inputs["images"],
            "series_mask": inputs["series_mask"].numpy().tolist(),
            "metadata_vectors": inputs["metadata"].numpy().tolist(),
        }

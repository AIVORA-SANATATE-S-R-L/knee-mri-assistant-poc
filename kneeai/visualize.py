"""Montage of the exact slices the model received, with the scores beside them."""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

TILE = 128
COLS = 8  # 8 of the 16 sampled slices per series


def save_montage(images_tensor, series_labels: List[str], scores: Dict[str, float],
                 thresholds: Dict[str, float], case_code: str, out_png: str, targets: List[str]) -> None:
    imgs = images_tensor[0].numpy()  # [S, N, 1, H, W]
    n_series = len(series_labels)
    label_w = 210
    panel_w = 360
    header_h = 46
    h = header_h + max(n_series, 1) * TILE
    w = label_w + COLS * TILE + panel_w
    canvas = np.full((h, w, 3), 255, np.uint8)

    cv2.putText(canvas, f"{case_code} - model input (8 of 16 sampled slices per series)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    for r in range(n_series):
        y0 = header_h + r * TILE
        cv2.putText(canvas, series_labels[r][:24], (8, y0 + TILE // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        n = imgs.shape[1]
        idx = np.linspace(0, n - 1, COLS).astype(int)
        for c, k in enumerate(idx):
            sl = (np.clip(imgs[r, k, 0], 0, 1) * 255).astype(np.uint8)
            sl = cv2.resize(sl, (TILE, TILE), interpolation=cv2.INTER_AREA)
            x0 = label_w + c * TILE
            canvas[y0:y0 + TILE, x0:x0 + TILE] = cv2.cvtColor(sl, cv2.COLOR_GRAY2BGR)

    px = label_w + COLS * TILE + 14
    y = header_h + 20
    cv2.putText(canvas, "Experimental model scores", (px, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    y += 24
    for t in targets:
        s = scores[t]
        th = thresholds[t]
        flag = ">= thr" if s >= th else "< thr"
        cv2.putText(canvas, f"{t:<17} {s:.3f} ({flag} {th:.2f})", (px, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        y += 22
    y += 10
    for line in ("Research demonstration only.", "Not a diagnosis. Not clinically", "validated. Scores are not", "calibrated probabilities."):
        cv2.putText(canvas, line, (px, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1, cv2.LINE_AA)
        y += 20
    cv2.imwrite(out_png, canvas)

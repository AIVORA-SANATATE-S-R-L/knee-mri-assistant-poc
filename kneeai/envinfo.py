"""Record the software and hardware environment of a test run."""

from __future__ import annotations

import datetime as dt
import platform
import subprocess
import sys
from typing import Any, Dict


def _git_head(path: str) -> str:
    try:
        head = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", path, "status", "--porcelain", "--untracked-files=no", "--", ".", ":!outputs"],
                                        text=True, stderr=subprocess.DEVNULL).strip()
        return head + ("-dirty (uncommitted code changes)" if dirty else "")
    except Exception:
        return "not a git checkout / git not available"


def collect(repo_root: str) -> Dict[str, Any]:
    import cv2
    import numpy
    import pandas
    import pydicom
    import torch
    import torchvision

    info = {
        "recorded_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "python_executable_arch": platform.architecture()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU inference)",
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "pydicom": pydicom.__version__,
        "opencv": cv2.__version__,
        "project_commit": _git_head(repo_root),
    }
    try:
        import psutil  # optional
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    return info

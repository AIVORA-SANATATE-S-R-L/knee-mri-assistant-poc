"""Index a DICOM study folder: group files by series and describe each series.

Works on any folder layout (flat export, one folder per series, nested PACS
export). Only headers are read here; pixel decoding is checked separately.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pydicom

LOCALIZER_WORDS = ("LOC", "SCOUT", "SURVEY", "PILOT", "3-PLANE", "3PLANE", "TRI-PILOT")


@dataclass
class SeriesInfo:
    series_uid: str
    series_number: Optional[int]
    description: str
    modality: str
    files: List[str] = field(default_factory=list)
    plane: str = "unknown"            # axial / coronal / sagittal / unknown
    fluid_sensitive: Optional[int] = None  # 1 / 0 / None (unknown)
    fat_suppressed: Optional[int] = None   # 1 / 0 / None (unknown)
    echo_time: Optional[float] = None
    repetition_time: Optional[float] = None
    is_localizer: bool = False
    is_derived: bool = False
    multiframe: bool = False
    transfer_syntax: str = ""
    rows: Optional[int] = None
    cols: Optional[int] = None
    laterality: str = ""
    burned_in_flag: str = ""
    sop_class: str = ""

    @property
    def n_files(self) -> int:
        return len(self.files)

    def short_label(self) -> str:
        p = {"axial": "AX", "coronal": "COR", "sagittal": "SAG"}.get(self.plane, "UNK")
        w = {1: "FLUID", 0: "NONFLUID"}.get(self.fluid_sensitive, "UNKW")
        f = {1: "FS", 0: "NOFS"}.get(self.fat_suppressed, "UNKFS")
        return f"{p}_{w}_{f}"


def _plane_from_iop(iop) -> str:
    try:
        v = [float(x) for x in iop]
        n = np.cross(np.array(v[0:3]), np.array(v[3:6]))
        axis = int(np.argmax(np.abs(n)))
        return ["sagittal", "coronal", "axial"][axis]
    except Exception:
        return "unknown"


def _text(ds, name: str) -> str:
    val = getattr(ds, name, "")
    if isinstance(val, (list, pydicom.multival.MultiValue)):
        return "\\".join(str(x) for x in val)
    return str(val) if val is not None else ""


def _classify(info: SeriesInfo, ds) -> None:
    desc = (info.description + " " + _text(ds, "ProtocolName") + " " + _text(ds, "SequenceName")).upper()
    image_type = _text(ds, "ImageType").upper()
    scan_opts = _text(ds, "ScanOptions").upper()

    info.plane = _plane_from_iop(getattr(ds, "ImageOrientationPatient", None)) if hasattr(ds, "ImageOrientationPatient") else "unknown"
    info.is_localizer = "LOCALIZER" in image_type or any(w in desc for w in LOCALIZER_WORDS)
    info.is_derived = "DERIVED" in image_type

    try:
        info.echo_time = float(ds.EchoTime) if hasattr(ds, "EchoTime") and ds.EchoTime not in (None, "") else None
    except Exception:
        info.echo_time = None
    try:
        info.repetition_time = float(ds.RepetitionTime) if hasattr(ds, "RepetitionTime") and ds.RepetitionTime not in (None, "") else None
    except Exception:
        info.repetition_time = None

    # Fat suppression: description keywords or scanner scan options
    tokens = desc.replace("-", " ").replace("_", " ").split()
    if any(w in desc for w in ("FAT SAT", "FATSAT", "SPAIR", "SPIR", "STIR", "TIRM", "DIXON W")) or "FS" in tokens or "FS" in scan_opts.replace("\\", " ").split() or "SFS" in scan_opts:
        info.fat_suppressed = 1
    elif tokens:
        info.fat_suppressed = 0

    # Fluid sensitivity: keywords first, echo time as fallback (TE >= 30 ms)
    if any(t in tokens for t in ("PD", "PDW", "DP", "T2", "STIR", "TIRM", "DESS", "MEDIC", "MERGE")) or any(w in desc for w in ("T2", "STIR", "TIRM")):
        info.fluid_sensitive = 1
    elif "T1" in desc:
        info.fluid_sensitive = 0
    elif info.echo_time is not None:
        info.fluid_sensitive = 1 if info.echo_time >= 30 else 0


def index_study(study_dir: str) -> Dict[str, SeriesInfo]:
    """Return {SeriesInstanceUID: SeriesInfo} for every readable DICOM image under study_dir."""
    series: Dict[str, SeriesInfo] = {}
    for root, _dirs, files in os.walk(study_dir):
        for name in files:
            if name.upper() == "DICOMDIR":
                continue
            path = os.path.join(root, name)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            except Exception:
                continue
            if not hasattr(ds, "SeriesInstanceUID") or not hasattr(ds, "SOPInstanceUID"):
                continue
            uid = str(ds.SeriesInstanceUID)
            info = series.get(uid)
            if info is None:
                try:
                    num = int(ds.SeriesNumber) if hasattr(ds, "SeriesNumber") and ds.SeriesNumber not in (None, "") else None
                except Exception:
                    num = None
                info = SeriesInfo(
                    series_uid=uid,
                    series_number=num,
                    description=_text(ds, "SeriesDescription"),
                    modality=_text(ds, "Modality"),
                )
                _classify(info, ds)
                info.multiframe = int(getattr(ds, "NumberOfFrames", 1) or 1) > 1
                try:
                    info.transfer_syntax = str(ds.file_meta.TransferSyntaxUID.name)
                except Exception:
                    info.transfer_syntax = ""
                info.rows = int(getattr(ds, "Rows", 0) or 0) or None
                info.cols = int(getattr(ds, "Columns", 0) or 0) or None
                info.laterality = _text(ds, "Laterality") or _text(ds, "ImageLaterality")
                info.burned_in_flag = _text(ds, "BurnedInAnnotation")
                info.sop_class = str(getattr(ds, "SOPClassUID", ""))
                series[uid] = info
            info.files.append(path)
    return series


def check_decodable(info: SeriesInfo) -> Optional[str]:
    """Decode the middle image of a series. Returns an error message or None if OK.

    The third-party loader silently replaces undecodable slices with black
    images, so this check is required before inference.
    """
    if not info.files:
        return "series has no files"
    path = sorted(info.files)[len(info.files) // 2]
    try:
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array
        if arr.size == 0 or not np.isfinite(arr.astype(np.float32)).all():
            return "decoded image is empty or non-finite"
        if float(arr.max()) == float(arr.min()):
            return "decoded image is constant (blank)"
    except Exception as exc:  # compression codec missing, corrupt file, etc.
        return f"pixel decode failed: {type(exc).__name__}: {exc}"
    return None

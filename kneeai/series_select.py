"""Choose which MRI series go into the model and stage them in the layout it expects.

The third-party loader takes the first six sub-folders it finds, in whatever
order the file system returns them. This module replaces that with an
explicit, logged selection rule so the radiographer can review it.

Selection rule (v0.1, to be reviewed by the radiographer):
  1. Exclude localizers/scouts, derived images, multi-frame objects and
     series with fewer than 8 images.
  2. Fill up to six slots, in this priority order:
       SAG fluid, COR fluid, AX fluid, SAG non-fluid, COR non-fluid, AX non-fluid
     For each slot choose the matching series with the most images
     (fat-suppressed preferred for fluid slots).
  3. If slots remain, add the largest remaining eligible series.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Dict, List, Tuple

import pandas as pd

from .dicom_index import SeriesInfo, check_decodable, index_study

SLOTS = [
    ("sagittal", 1), ("coronal", 1), ("axial", 1),
    ("sagittal", 0), ("coronal", 0), ("axial", 0),
]
MIN_IMAGES = 8


def eligibility(info: SeriesInfo) -> str:
    if info.modality and info.modality.upper() != "MR":
        return f"not MR ({info.modality})"
    if info.is_localizer:
        return "localizer/scout"
    if info.is_derived:
        return "derived image"
    if info.multiframe:
        return "multi-frame object (not supported by loader)"
    if info.n_files < MIN_IMAGES:
        return f"too few images ({info.n_files})"
    return "eligible"


def select_series(series: Dict[str, SeriesInfo], max_series: int = 6) -> Tuple[List[SeriesInfo], pd.DataFrame]:
    rows = []
    eligible = []
    for info in series.values():
        status = eligibility(info)
        rows.append({
            "series_number": info.series_number,
            "description": info.description,
            "plane": info.plane,
            "fluid_sensitive": info.fluid_sensitive,
            "fat_suppressed": info.fat_suppressed,
            "echo_time_ms": info.echo_time,
            "n_images": info.n_files,
            "matrix": f"{info.rows}x{info.cols}",
            "transfer_syntax": info.transfer_syntax,
            "eligibility": status,
            "selected_slot": "",
            "series_uid": info.series_uid,
        })
        if status == "eligible":
            eligible.append(info)

    chosen: List[SeriesInfo] = []
    slot_of: Dict[str, str] = {}
    for plane, fluid in SLOTS:
        cands = [s for s in eligible if s.plane == plane and s.fluid_sensitive == fluid and s not in chosen]
        if not cands:
            continue
        cands.sort(key=lambda s: ((s.fat_suppressed == 1) if fluid == 1 else 0, s.n_files), reverse=True)
        chosen.append(cands[0])
        slot_of[cands[0].series_uid] = f"{ {'sagittal': 'SAG', 'coronal': 'COR', 'axial': 'AX'}[plane]}-{'fluid' if fluid else 'nonfluid'}"
        if len(chosen) >= max_series:
            break
    for s in sorted(eligible, key=lambda s: s.n_files, reverse=True):
        if len(chosen) >= max_series:
            break
        if s not in chosen:
            chosen.append(s)
            slot_of[s.series_uid] = "fill"

    table = pd.DataFrame(rows)
    if not table.empty:
        table["selected_slot"] = table["series_uid"].map(slot_of).fillna("")
        table = table.sort_values(["series_number"], na_position="last").reset_index(drop=True)
    return chosen, table


def _link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)  # hard link: no second copy of patient data on disk
    except OSError:
        shutil.copy2(src, dst)


def stage_study(study_dir: str, case_code: str, work_root=None, max_series: int = 6, copy_files: bool = True):
    """Index, select and stage the chosen series as <tmp>/case_code/<Sk_LABEL>/NNNN.dcm.

    work_root=None uses a fresh temporary folder next to the study (same disk, so
    hard links work). The caller removes os.path.dirname(staged) afterwards.
    copy_files=False only indexes, selects and checks decoding (for `inspect`).
    Returns (staged_study_dir, metadata_df, selection_table, chosen_series, errors).
    """
    series = index_study(study_dir)
    chosen, table = select_series(series, max_series=max_series)
    errors = []
    if not chosen:
        errors.append("no eligible MR series found")

    staged = None
    if copy_files:
        if work_root is None:
            work_root = tempfile.mkdtemp(prefix="kneeai_stage_", dir=os.path.dirname(os.path.abspath(study_dir)))
        else:
            work_root = tempfile.mkdtemp(prefix="kneeai_stage_", dir=work_root)
        staged = os.path.join(work_root, case_code)
        os.makedirs(staged)

    meta_rows = []
    for k, info in enumerate(chosen, start=1):
        err = check_decodable(info)
        if err:
            errors.append(f"series {info.series_number} '{info.description}': {err}")
        folder = f"S{k}_{info.short_label()}"
        if staged:
            dst = os.path.join(staged, folder)
            os.makedirs(dst)
            for i, src in enumerate(sorted(info.files), start=1):
                _link_or_copy(src, os.path.join(dst, f"{i:04d}.dcm"))
        meta_rows.append({
            "StudyInstanceUID": case_code,
            "SeriesInstanceUID": folder,
            "Anatomical_Plane": info.plane if info.plane != "unknown" else "",
            "Fluid_Sensitive": info.fluid_sensitive if info.fluid_sensitive is not None else -1,
            "Fat_Suppression": info.fat_suppressed if info.fat_suppressed is not None else -1,
            "source_series_number": info.series_number,
            "source_description": info.description,
            "n_images": info.n_files,
        })
    return staged, pd.DataFrame(meta_rows), table, chosen, errors

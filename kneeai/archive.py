"""Find and extract single knee MRI studies from a large mixed DICOM archive.

A clinic export often holds a whole year of examinations (tens of thousands of
files, all patients mixed). `find_studies` reads only the headers once and
groups files by study; `extract_study` copies one chosen study's original
files, unchanged, into its own folder for `deid`.

The study list with patient names is written ONLY to a local CSV chosen by the
user (keep it outside the repository and outside cloud-synced folders). The
terminal shows study numbers, dates and descriptions, no names.
"""

from __future__ import annotations

import csv
import os
import shutil
import time
from collections import defaultdict
from typing import Dict, List, Optional

import pydicom

KNEE_WORDS = ("KNEE", "GENU", "GENUNCHI", "ΓΟΝΑΤ", "GONAT", "KNIE", "GENOU", "GINOCCHIO", "RODILLA")
TAGS = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "Modality", "StudyDate", "StudyTime",
        "StudyDescription", "SeriesDescription", "BodyPartExamined", "ProtocolName", "PatientName",
        "PatientID", "SeriesNumber"]


def _is_knee(text: str) -> bool:
    t = text.upper()
    return any(w in t for w in KNEE_WORDS)


def find_studies(archive: str, index_csv: str, knee_only: bool = True, progress_every: int = 2000) -> List[dict]:
    """Scan `archive` recursively and write one row per study to `index_csv`.

    A companion file `<index_csv>.files.csv` maps every file to its study, so
    extraction does not need to scan again.
    """
    studies: Dict[str, dict] = {}
    files_of: Dict[str, List[str]] = defaultdict(list)
    n_seen = n_dicom = 0
    t0 = time.time()
    for root, _dirs, files in os.walk(archive):
        for name in files:
            n_seen += 1
            if progress_every and n_seen % progress_every == 0:
                print(f"  scanned {n_seen} files, {len(studies)} studies so far ({time.time() - t0:.0f}s)", flush=True)
            path = os.path.join(root, name)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True, specific_tags=TAGS)
            except Exception:
                continue
            uid = str(getattr(ds, "StudyInstanceUID", "") or "")
            if not uid or not getattr(ds, "SOPInstanceUID", None):
                continue
            n_dicom += 1
            st = studies.get(uid)
            if st is None:
                st = studies[uid] = {
                    "study_uid": uid,
                    "date": str(getattr(ds, "StudyDate", "") or ""),
                    "time": str(getattr(ds, "StudyTime", "") or "")[:6],
                    "description": str(getattr(ds, "StudyDescription", "") or ""),
                    "patient_name": str(getattr(ds, "PatientName", "") or ""),
                    "patient_id": str(getattr(ds, "PatientID", "") or ""),
                    "modalities": set(), "series": {}, "body_text": "",
                }
            st["modalities"].add(str(getattr(ds, "Modality", "") or ""))
            sdesc = str(getattr(ds, "SeriesDescription", "") or "")
            suid = str(getattr(ds, "SeriesInstanceUID", "") or "")
            if suid not in st["series"]:
                st["series"][suid] = [sdesc, 0]
                st["body_text"] += " " + " ".join(str(getattr(ds, t, "") or "") for t in
                                                   ("BodyPartExamined", "ProtocolName")) + " " + sdesc
            st["series"][suid][1] += 1
            files_of[uid].append(path)

    rows = []
    for st in sorted(studies.values(), key=lambda s: (s["date"], s["time"])):
        knee = _is_knee(st["description"] + " " + st["body_text"])
        if knee_only and (not knee or "MR" not in st["modalities"]):
            continue
        series_list = sorted(st["series"].values(), key=lambda x: x[0])
        rows.append({
            "study_no": len(rows) + 1,
            "date": st["date"], "time": st["time"], "description": st["description"],
            "modalities": "/".join(sorted(m for m in st["modalities"] if m)),
            "n_series": len(series_list), "n_images": sum(n for _d, n in series_list),
            "series": "; ".join(f"{d or '?'} ({n})" for d, n in series_list),
            "patient_name": st["patient_name"], "patient_id": st["patient_id"], "study_uid": st["study_uid"],
        })

    os.makedirs(os.path.dirname(os.path.abspath(index_csv)) or ".", exist_ok=True)
    with open(index_csv, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["study_no"])
        w.writeheader()
        w.writerows(rows)
    keep = {r["study_uid"] for r in rows}
    with open(index_csv + ".files.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["study_uid", "path"])
        for uid in keep:
            for p in files_of[uid]:
                w.writerow([uid, p])
    print(f"Scanned {n_seen} files ({n_dicom} DICOM) in {time.time() - t0:.0f}s; "
          f"{len(studies)} studies, {len(rows)} {'knee MR ' if knee_only else ''}studies listed.")
    return rows


def extract_study(index_csv: str, study_no: int, out_dir: str) -> int:
    """Copy the original files of study `study_no` (from find_studies) into out_dir, unchanged."""
    with open(index_csv, encoding="utf-8-sig") as fh:
        rows = {int(r["study_no"]): r for r in csv.DictReader(fh)}
    if study_no not in rows:
        raise ValueError(f"Study number {study_no} not in {index_csv} (1..{max(rows) if rows else 0}).")
    uid = rows[study_no]["study_uid"]
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f"Output folder is not empty: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with open(index_csv + ".files.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["study_uid"] == uid:
                n += 1
                shutil.copy2(r["path"], os.path.join(out_dir, f"{n:05d}.dcm"))
    return n

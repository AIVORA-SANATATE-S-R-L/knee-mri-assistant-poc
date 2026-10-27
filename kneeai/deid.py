"""Pseudonymisation and identifier check for DICOM studies.

Intended to be run ON THE DATA PROVIDER'S (ELDA) COMPUTER, before any data
leaves their control. The re-identification key file stays with ELDA.

This is a basic profile for an internal research demonstration. It does not
detect text burned into pixels: every exported study and screenshot must
still be reviewed visually.
"""

from __future__ import annotations

import csv
import hashlib
import os
from typing import Dict, List, Tuple

import pydicom
from pydicom.uid import generate_uid

from .dicom_index import index_study

# Attributes that identify the patient, staff, institution or the visit.
IDENTIFYING_TAGS = [
    "PatientName", "PatientID", "PatientBirthDate", "PatientBirthTime", "PatientAddress",
    "PatientTelephoneNumbers", "OtherPatientIDs", "OtherPatientNames", "OtherPatientIDsSequence",
    "PatientMotherBirthName", "MilitaryRank", "EthnicGroup", "Occupation", "AdditionalPatientHistory",
    "PatientComments", "MedicalRecordLocator", "InsurancePlanIdentification",
    "AccessionNumber", "StudyID", "IssuerOfPatientID",
    "ReferringPhysicianName", "ReferringPhysicianAddress", "ReferringPhysicianTelephoneNumbers",
    "PerformingPhysicianName", "OperatorsName", "PhysiciansOfRecord", "RequestingPhysician",
    "NameOfPhysiciansReadingStudy", "ScheduledPerformingPhysicianName",
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "StationName", "DeviceSerialNumber",
    "RequestAttributesSequence", "ReferencedPatientSequence", "ReferencedStudySequence",
    "RequestedProcedureID", "ScheduledProcedureStepID", "PerformedProcedureStepID",
    "PerformedProcedureStepDescription", "StudyComments", "ImageComments",
]

DATE_TIME_TAGS = [
    "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate", "InstanceCreationDate",
    "PerformedProcedureStepStartDate", "StudyTime", "SeriesTime", "AcquisitionTime",
    "ContentTime", "InstanceCreationTime", "PerformedProcedureStepStartTime",
    "AcquisitionDateTime",
]

UID_TAGS = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID"]

# Non-image objects that often contain names or report text (screenshots, dose reports, reports)
EXCLUDED_MODALITIES = {"SR", "OT", "DOC", "PR", "KO", "REG"}


def find_identifiers(study_dir: str) -> List[Tuple[str, str, str]]:
    """Return a list of (file, attribute, 'present') for identifying data still in the study."""
    findings = []
    for root, _d, files in os.walk(study_dir):
        for name in files:
            path = os.path.join(root, name)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            except Exception:
                continue
            if not hasattr(ds, "SOPInstanceUID"):
                continue
            for tag in IDENTIFYING_TAGS:
                if tag in ds and str(ds.data_element(tag).value).strip() not in ("", "None"):
                    val = str(ds.data_element(tag).value).strip()
                    if tag in ("PatientName", "PatientID") and val.startswith(os.path.basename(study_dir.rstrip("/\\"))):
                        continue  # already replaced by the case code
                    findings.append((path, tag, "present"))
            for tag in DATE_TIME_TAGS:
                if tag in ds and str(ds.data_element(tag).value).strip():
                    findings.append((path, tag, "present"))
            has_private = any(el.tag.is_private for el in ds)
            if has_private:
                findings.append((path, "private tags", "present"))
            if str(getattr(ds, "BurnedInAnnotation", "")).upper() == "YES":
                findings.append((path, "BurnedInAnnotation", "YES"))
            if str(getattr(ds, "Modality", "")).upper() in EXCLUDED_MODALITIES:
                findings.append((path, f"Modality {ds.Modality}", "non-image object"))
    return findings


def _new_uid(original: str, salt: str) -> str:
    return generate_uid(entropy_srcs=[salt, original])


def pseudonymise_study(src_dir: str, out_root: str, case_code: str, salt: str, key_file: str) -> Dict[str, int]:
    """Copy one study to out_root/case_code with identifiers removed.

    - names / IDs replaced by the case code
    - staff, institution, device and visit identifiers removed
    - dates and times removed (patient age and sex kept)
    - private tags removed
    - UIDs replaced by new UIDs derived from a secret salt
    - screenshots, structured reports and other non-image objects skipped
    - original StudyInstanceUID -> case code written to key_file (keep at ELDA)
    """
    if not salt or len(salt) < 12:
        raise ValueError("Use a secret salt of at least 12 characters and keep it at ELDA.")
    out_dir = os.path.join(out_root, case_code)
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f"Output folder already exists and is not empty: {out_dir}")

    series = index_study(src_dir)
    counts = {"series": 0, "files": 0, "skipped_non_image": 0}
    original_study_uids = set()

    for s_idx, info in enumerate(sorted(series.values(), key=lambda s: (s.series_number or 0, s.series_uid)), start=1):
        if info.modality.upper() in EXCLUDED_MODALITIES:
            counts["skipped_non_image"] += info.n_files
            continue
        s_dir = os.path.join(out_dir, f"series_{s_idx:02d}")
        os.makedirs(s_dir, exist_ok=True)
        counts["series"] += 1
        for f_idx, path in enumerate(sorted(info.files), start=1):
            ds = pydicom.dcmread(path, force=True)
            original_study_uids.add(str(getattr(ds, "StudyInstanceUID", "")))
            for tag in IDENTIFYING_TAGS + DATE_TIME_TAGS:
                if tag in ds:
                    delattr(ds, tag)
            ds.remove_private_tags()
            ds.PatientName = case_code
            ds.PatientID = case_code
            for tag in UID_TAGS:
                if tag in ds:
                    setattr(ds, tag, _new_uid(str(ds.data_element(tag).value), salt))
            if hasattr(ds, "file_meta") and "SOPInstanceUID" in ds:
                ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            ds.PatientIdentityRemoved = "YES"
            ds.DeidentificationMethod = "kneeai basic pseudonymisation profile v0.1"
            ds.save_as(os.path.join(s_dir, f"{f_idx:04d}.dcm"))
            counts["files"] += 1

    new_file = not os.path.exists(key_file)
    os.makedirs(os.path.dirname(os.path.abspath(key_file)), exist_ok=True)
    with open(key_file, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["case_code", "original_StudyInstanceUID", "source_folder_sha1"])
        for uid in sorted(original_study_uids):
            w.writerow([case_code, uid, hashlib.sha1(os.path.abspath(src_dir).encode()).hexdigest()])
    return counts

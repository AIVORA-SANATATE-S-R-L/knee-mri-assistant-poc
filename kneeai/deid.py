"""Pseudonymisation and identifier check for DICOM studies.

Intended to be run ON THE DATA PROVIDER'S (ELDA) COMPUTER, before any data
leaves their control. The re-identification key file stays with ELDA.

Profile (v0.2, value-representation based, applied at every nesting depth):
  * person names (PN) removed; PatientName/PatientID set to the case code
  * dates and times (DA/DT/TM) removed; age and sex kept
  * free text (LT/ST/UT) removed
  * listed identifying attributes (institution, accession, device...) removed
  * private tags removed
  * every non-standard UID (UI) remapped consistently with a secret salt, so
    internal references stay valid but cannot be linked back to the PACS
It does not detect text burned into pixels: every exported study and
screenshot must still be reviewed visually.
"""

from __future__ import annotations

import csv
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

# Non-image objects that often contain names or report text (screenshots, dose reports, reports)
EXCLUDED_MODALITIES = {"SR", "OT", "DOC", "PR", "KO", "REG"}


STANDARD_UID_ROOT = "1.2.840.10008."  # DICOM-defined UIDs (SOP classes, transfer syntaxes): keep
TEXT_VRS = {"LT", "ST", "UT"}
DATE_VRS = {"DA", "DT", "TM"}
PROFILE_NAME = "kneeai pseudonymisation profile v0.2"


def _iter_elements(ds):
    """Yield every data element at every nesting depth."""
    for elem in ds:
        yield elem
        if elem.VR == "SQ":
            for item in elem.value or []:
                yield from _iter_elements(item)


def find_identifiers(study_dir: str) -> List[Tuple[str, str, str]]:
    """Return (file, attribute, finding) for anything identifying left in the study.

    Searches nested sequences too. A file not processed by `deid` is always
    reported, because original UIDs cannot be recognised from their value.
    """
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
            case_code = str(getattr(ds, "PatientID", ""))
            processed = str(getattr(ds, "PatientIdentityRemoved", "")).upper() == "YES" and \
                str(getattr(ds, "DeidentificationMethod", "")).startswith(PROFILE_NAME)
            if not processed:
                findings.append((path, "PatientIdentityRemoved/DeidentificationMethod", "not processed by kneeai deid"))
            for elem in _iter_elements(ds):
                kw = elem.keyword or str(elem.tag)
                val = str(elem.value).strip() if elem.VR != "SQ" else ""
                if elem.tag.is_private:
                    findings.append((path, f"private tag {elem.tag}", "present"))
                elif elem.VR == "PN" and val and val != case_code:
                    findings.append((path, kw, "person name"))
                elif elem.VR in DATE_VRS and val:
                    findings.append((path, kw, "date/time"))
                elif elem.VR in TEXT_VRS and val:
                    findings.append((path, kw, "free text"))
                elif kw in IDENTIFYING_TAGS and val and val != case_code:
                    findings.append((path, kw, "identifier"))
            if str(getattr(ds, "BurnedInAnnotation", "")).upper() == "YES":
                findings.append((path, "BurnedInAnnotation", "YES"))
            if str(getattr(ds, "Modality", "")).upper() in EXCLUDED_MODALITIES:
                findings.append((path, f"Modality {ds.Modality}", "non-image object"))
    return findings


def _new_uid(original: str, salt: str) -> str:
    return generate_uid(entropy_srcs=[salt, original])


def _scrub(ds, salt: str) -> None:
    """Apply the profile in place, recursively."""
    ds.remove_private_tags()

    def cb(dataset, elem):
        kw = elem.keyword or ""
        if elem.VR in ("PN",) or elem.VR in DATE_VRS or elem.VR in TEXT_VRS or kw in IDENTIFYING_TAGS:
            del dataset[elem.tag]
        elif elem.VR == "UI" and elem.value:
            vals = elem.value if isinstance(elem.value, (list, pydicom.multival.MultiValue)) else [elem.value]
            new = [v if str(v).startswith(STANDARD_UID_ROOT) else _new_uid(str(v), salt) for v in vals]
            elem.value = new if len(new) > 1 else new[0]

    ds.walk(cb)


def _cases_in_key(key_file: str) -> set:
    if not os.path.exists(key_file):
        return set()
    with open(key_file, encoding="utf-8") as fh:
        return {row[0] for row in csv.reader(fh) if row}


def pseudonymise_study(src_dir: str, out_root: str, case_code: str, salt: str, key_file: str) -> Dict[str, int]:
    """Copy one study to out_root/case_code with the profile above applied.

    Screenshots, structured reports and other non-image objects are skipped.
    The original StudyInstanceUID -> case code mapping goes to key_file, which
    stays with the data provider.
    """
    if not salt or len(salt) < 12:
        raise ValueError("Use a secret phrase (salt) of at least 12 characters and keep it with the data provider.")
    out_dir = os.path.join(out_root, case_code)
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f"Output folder already exists and is not empty: {out_dir}")
    if case_code in _cases_in_key(key_file):
        raise ValueError(f"Case code {case_code} is already used in {key_file}; choose a new code.")

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
            _scrub(ds, salt)
            ds.PatientName = case_code
            ds.PatientID = case_code
            if hasattr(ds, "file_meta") and "SOPInstanceUID" in ds:
                ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            ds.PatientIdentityRemoved = "YES"
            ds.DeidentificationMethod = PROFILE_NAME
            ds.save_as(os.path.join(s_dir, f"{f_idx:04d}.dcm"))
            counts["files"] += 1

    new_file = not os.path.exists(key_file)
    os.makedirs(os.path.dirname(os.path.abspath(key_file)), exist_ok=True)
    with open(key_file, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["case_code", "original_StudyInstanceUID"])
        for uid in sorted(original_study_uids):
            w.writerow([case_code, uid])
    return counts

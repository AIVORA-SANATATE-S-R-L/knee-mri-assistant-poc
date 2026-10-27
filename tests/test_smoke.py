"""End-to-end smoke test on SYNTHETIC DICOM data (no patient data).

Creates a fake knee study with fake identifiers, pseudonymises it, inspects it,
runs the model twice and builds a report – all under outputs/smoke_test/.
Proves only that the software executes; the scores are meaningless.

    python -m kneeai smoke-test      or      python -m pytest tests
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SERIES = [  # number, description, IOP, n images, TE
    (1, "localizer", [0, 1, 0, 0, 0, -1], 3, 5.0),
    (2, "PD_TSE_FS_SAG", [0, 1, 0, 0, 0, -1], 24, 35.0),
    (3, "T1_TSE_COR", [1, 0, 0, 0, 0, -1], 22, 12.0),
    (4, "PD_TSE_FS_AX", [1, 0, 0, 0, 1, 0], 26, 33.0),
]


def make_synthetic_study(folder: str) -> None:
    rng = np.random.default_rng(0)
    study_uid = generate_uid()
    for num, desc, iop, n, te in SERIES:
        series_uid = generate_uid()
        for k in range(n):
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = MRImageStorage
            sop = generate_uid()
            meta.MediaStorageSOPInstanceUID = sop
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
            ds.SOPClassUID = MRImageStorage
            ds.SOPInstanceUID = sop
            ds.StudyInstanceUID = study_uid
            ds.SeriesInstanceUID = series_uid
            ds.Modality = "MR"
            ds.PatientName = "TEST^SYNTHETIC"
            ds.PatientID = "FAKE123"
            ds.PatientBirthDate = "19800101"
            ds.InstitutionName = "Synthetic Hospital"
            ds.StudyDate = "20260101"
            ds.AccessionNumber = "ACC0001"
            ds.SeriesNumber = num
            ds.InstanceNumber = k + 1
            ds.SeriesDescription = desc
            ds.EchoTime = te
            ds.ImageOrientationPatient = iop
            normal = np.cross(iop[:3], iop[3:])
            ds.ImagePositionPatient = list((normal * k * 3.0).astype(float))
            ds.Rows = ds.Columns = 64
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            yy, xx = np.mgrid[0:64, 0:64]
            img = 400 + 300 * np.exp(-((xx - 32) ** 2 + (yy - 32 - k % 5) ** 2) / 200) + rng.normal(0, 20, (64, 64))
            ds.PixelData = np.clip(img, 0, 4000).astype(np.uint16).tobytes()
            ds.add_new(0x00191010, "LO", "private vendor data")
            # deliberately unordered, flat file names like many PACS exports
            ds.save_as(os.path.join(folder, f"IMG_{num}_{(k * 7) % n:03d}_{generate_uid()[-6:]}.dcm"), enforce_file_format=True)


def run_smoke() -> None:
    from kneeai import cli
    base = os.path.join(ROOT, "outputs", "smoke_test")
    shutil.rmtree(base, ignore_errors=True)
    raw = os.path.join(base, "raw_synthetic", "study")
    os.makedirs(raw)
    make_synthetic_study(raw)

    cli.OUT = os.path.join(base, "outputs")
    cli.WORK = os.path.join(base, "_staged")
    deid_root = os.path.join(base, "pseudonymised")
    cli.main(["deid", "--input", raw, "--output", deid_root, "--case", "SYN-001",
              "--salt", "synthetic-test-salt-0001", "--key-file", os.path.join(base, "key_SYNTHETIC.csv")])
    study = os.path.join(deid_root, "SYN-001")

    from kneeai.deid import find_identifiers
    assert not find_identifiers(study), "identifiers left after pseudonymisation"

    labels = os.path.join(base, "labels.csv")
    pd.DataFrame([{"case_code": "SYN-001", "ACL": "0", "MCL": "0", "Medial Meniscus": "U", "Lateral Meniscus": "0"}]).to_csv(labels, index=False)

    cli.main(["inspect", "--study", study, "--case", "SYN-001"])
    cli.main(["env"])
    cli.main(["verify-model"])
    cli.main(["run", "--study", study, "--case", "SYN-001", "--labels", labels, "--run-label", "run1"])
    cli.main(["run", "--study", study, "--case", "SYN-001", "--labels", labels, "--run-label", "run2"])
    # the original (identifiable) study must be refused
    cli.main(["run", "--study", raw, "--case", "SYN-RAW", "--labels", labels, "--run-label", "refuse_check"])

    res = pd.read_csv(os.path.join(cli.OUT, "results.csv"))
    ok = res[(res["case_code"] == "SYN-001") & res["error"].isna()]
    assert len(ok) == 8, f"expected 8 result rows, got {len(ok)}"
    assert ok.groupby("target")["score"].agg(lambda s: s.max() - s.min()).max() < 1e-4, "runs not reproducible"
    refused = res[res["case_code"] == "SYN-RAW"]
    assert refused["error"].str.contains("identifiers").all(), "identifiable study was not refused"
    sel = pd.read_csv(os.path.join(cli.OUT, "SYN-001", "selection.csv"))
    assert (sel["selected_slot"].fillna("") != "").sum() == 3, "expected 3 selected series (localizer excluded)"

    cli.main(["report"])
    print("\nSMOKE TEST PASSED (synthetic data – proves the software executes, nothing more)")


def test_smoke():
    run_smoke()


if __name__ == "__main__":
    run_smoke()

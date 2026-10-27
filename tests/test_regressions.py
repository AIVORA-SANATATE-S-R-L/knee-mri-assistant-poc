"""Regression tests for problems found in the 2026-10-27 architecture review.

Each test builds a synthetic study that reproduces one failure mode.
"""

from __future__ import annotations

import os
import shutil
import sys

import pydicom
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kneeai.deid import find_identifiers, pseudonymise_study  # noqa: E402
from kneeai.dicom_index import index_study  # noqa: E402
from kneeai.series_select import stage_study  # noqa: E402
from tests.test_smoke import make_synthetic_study  # noqa: E402

SALT = "regression-test-salt-01"


@pytest.fixture()
def raw(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    make_synthetic_study(str(d))
    return str(d)


def _edit_all(folder, fn):
    for f in sorted(os.listdir(folder)):
        p = os.path.join(folder, f)
        ds = pydicom.dcmread(p)
        if fn(ds, f) is not False:
            ds.save_as(p)


def test_nested_identifiers_and_uid_references_removed(raw, tmp_path):
    originals = set()

    def add_nested(ds, _f):
        originals.add(str(ds.SOPInstanceUID))
        ref = Dataset()
        ref.ReferencedSOPClassUID = ds.SOPClassUID
        ref.ReferencedSOPInstanceUID = ds.SOPInstanceUID
        ds.ReferencedImageSequence = Sequence([ref])
        proc = Dataset()
        proc.CodeMeaning = "MRI KNEE"
        proc.ScheduledPerformingPhysicianName = "DR^NESTED"
        proc.ScheduledProcedureStepStartDate = "20260102"
        ds.ProcedureCodeSequence = Sequence([proc])
        ds.ImageComments = "patient John, follow-up"

    _edit_all(raw, add_nested)
    # an unprocessed study must be flagged even though top-level tags look clean
    assert find_identifiers(raw)
    out = tmp_path / "pseudo"
    pseudonymise_study(raw, str(out), "R-001", SALT, str(tmp_path / "key.csv"))
    study = out / "R-001"
    assert find_identifiers(str(study)) == []
    for root, _d, files in os.walk(study):
        for f in files:
            text = str(pydicom.dcmread(os.path.join(root, f)))
            assert "NESTED" not in text and "John" not in text and "20260102" not in text
            assert not any(u in text for u in originals), "original UID still referenced"


def test_case_code_reuse_refused(raw, tmp_path):
    key = str(tmp_path / "key.csv")
    pseudonymise_study(raw, str(tmp_path / "a"), "R-002", SALT, key)
    with pytest.raises(ValueError):
        pseudonymise_study(raw, str(tmp_path / "b"), "R-002", SALT, key)


@pytest.mark.parametrize("desc", ["AX PDFS", "ax_pd_fs", "T2 FATSAT AX", "PDW SPAIR AX", "ax pd-fs tse"])
def test_fat_sat_and_fluid_variants(raw, desc):
    def rename(ds, _f):
        if ds.SeriesNumber == 4:
            ds.SeriesDescription = desc
            ds.EchoTime = 10.0  # force the keywords, not the TE fallback, to decide
        else:
            return False

    _edit_all(raw, rename)
    s4 = [s for s in index_study(raw).values() if s.series_number == 4][0]
    assert s4.fat_suppressed == 1 and s4.fluid_sensitive == 1, (desc, s4.fat_suppressed, s4.fluid_sensitive)


def test_mixed_orientations_in_one_series_are_split(raw):
    uid = {}

    def merge(ds, _f):
        if ds.SeriesNumber == 2:
            uid.setdefault("u", str(ds.SeriesInstanceUID))
        return False

    _edit_all(raw, merge)

    def relabel(ds, _f):
        if ds.SeriesNumber == 3:
            ds.SeriesInstanceUID = uid["u"]
            ds.SeriesNumber = 2
        else:
            return False

    _edit_all(raw, relabel)
    stacks = [s for s in index_study(raw).values() if s.series_number == 2]
    assert sorted((s.plane, s.n_files) for s in stacks) == [("coronal", 22), ("sagittal", 24)]


def test_corrupt_slices_are_reported(raw, tmp_path):
    out = tmp_path / "pseudo"
    pseudonymise_study(raw, str(out), "R-003", SALT, str(tmp_path / "key.csv"))
    study = out / "R-003"
    series_dir = sorted(p for p in study.iterdir())[1]
    for f in sorted(series_dir.iterdir())[:6]:
        ds = pydicom.dcmread(f)
        ds.PixelData = ds.PixelData[:100]
        ds.save_as(f)
    staged, _meta, _t, _c, errors = stage_study(str(study), "R-003")
    shutil.rmtree(os.path.dirname(staged), ignore_errors=True)
    assert any("cannot be decoded" in e for e in errors), errors


def test_staging_leaves_nothing_behind(raw, tmp_path):
    out = tmp_path / "pseudo"
    pseudonymise_study(raw, str(out), "R-004", SALT, str(tmp_path / "key.csv"))
    before = set(os.listdir(out))
    staged, *_ = stage_study(str(out / "R-004"), "R-004")
    shutil.rmtree(os.path.dirname(staged))
    assert set(os.listdir(out)) == before

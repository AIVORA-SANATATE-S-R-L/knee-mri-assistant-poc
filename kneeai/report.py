"""Build the dated technical demonstration report from the recorded runs."""

from __future__ import annotations

import datetime as dt
import json
import os
from typing import List

import pandas as pd

from .engine import PINNED_COMMIT, PINNED_SHA256, PINNED_SIZE, TARGETS_REPORTED

TOLERANCE = 1e-4  # max allowed score difference between repeated runs


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(none)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines) + "\n"


def build_report(outputs_dir: str) -> str:
    res_path = os.path.join(outputs_dir, "results.csv")
    if not os.path.exists(res_path):
        raise FileNotFoundError("No results.csv yet. Run `python -m kneeai run ...` first.")
    df = pd.read_csv(res_path, dtype={"radiologist_reference": str})
    env = {}
    env_path = os.path.join(outputs_dir, "environment.json")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as fh:
            env = json.load(fh)
    model_check = {}
    mc_path = os.path.join(outputs_dir, "model_check.json")
    if os.path.exists(mc_path):
        with open(mc_path, encoding="utf-8") as fh:
            model_check = json.load(fh)

    ok = df[df["error"].fillna("") == ""]
    failed = df[df["error"].fillna("") != ""][["run_label", "case_code", "error"]].drop_duplicates()

    # Results table: last run per case
    last_run = ok.sort_values("run_timestamp").groupby("case_code")["run_label"].last()
    latest = ok[ok.apply(lambda r: last_run.get(r["case_code"]) == r["run_label"], axis=1)]
    results_tbl = latest[["case_code", "target", "score", "threshold", "predicted_label",
                          "radiologist_reference", "agreement"]].copy()
    results_tbl["score"] = results_tbl["score"].map(lambda x: f"{x:.4f}")

    # Reproducibility: spread of scores across runs for each case/target
    rep = ok.groupby(["case_code", "target"])["score"].agg(["count", "min", "max"]).reset_index()
    rep["max_difference"] = (rep["max"] - rep["min"]).map(lambda x: f"{x:.2e}")
    rep_cases = rep[rep["count"] >= 2]
    reproducible = (not rep_cases.empty) and float((rep_cases["max"] - rep_cases["min"]).max()) <= TOLERANCE
    rep_summary = rep_cases.groupby("case_code").agg(runs=("count", "max"),
                                                     max_difference=("max_difference", "max")).reset_index()

    agree = latest[latest["agreement"].isin(["agree", "disagree"])]
    n_agree = int((agree["agreement"] == "agree").sum())
    n_dis = int((agree["agreement"] == "disagree").sum())

    series_tbl = latest[["case_code", "selected_series", "n_series_used", "total_seconds"]].drop_duplicates("case_code")

    today = dt.date.today().isoformat()
    first = df["run_timestamp"].min()
    last = df["run_timestamp"].max()

    checks = [
        ("Full checkpoint downloaded (not an LFS pointer)", model_check.get("size_bytes") == PINNED_SIZE),
        ("Checkpoint SHA-256 matches pinned value", model_check.get("sha256") == PINNED_SHA256),
        ("Safe loading with weights_only=True", bool(model_check.get("safe_load_ok"))),
        ("Strict key match (no missing/unexpected keys)", bool(model_check.get("strict_keys_ok"))),
        ("Finite outputs on all completed cases", bool(len(ok)) and bool(ok["score"].notna().all())),
        ("Repeated runs identical within tolerance 1e-4", reproducible),
        ("No failed cases", failed.empty),
    ]
    check_md = "\n".join(f"| {name} | {'PASSED' if val else 'NOT MET / PENDING'} |" for name, val in checks)

    md = f"""# Technical demonstration report supporting TRL assessment

**Project:** AI assistant for knee MRI (PoCIDIF 2021–2027, Action 2.1)
**Report date:** {today}
**Test period:** {first} to {last}
**Prepared by:** [NAME, ROLE]  **Reviewed by:** [NAME, ROLE]

> Research demonstration only. Model scores are experimental outputs, not diagnoses and not calibrated
> clinical probabilities. This report does not certify a TRL level and does not claim clinical validation.

## 1. Purpose

Show, in a laboratory setting, that the integrated prototype pipeline (data preparation → series selection
→ model execution → output reporting) runs end-to-end on complete MRI studies from the clinical partner,
reproducibly and with traceable versions.

## 2. What is pre-existing and what was developed by the team

| Component | Origin |
|---|---|
| Neural network architecture and weights (`rsna-knee-mri-model`, commit `{PINNED_COMMIT[:12]}`) | Third-party, MIT-licensed repository. Not developed by the team. |
| Pseudonymisation and identifier check (`kneeai/deid.py`, profile v0.2) | Developed by the team |
| Study indexing, sequence classification and explicit series selection (`kneeai/dicom_index.py`, `kneeai/series_select.py`) | Developed by the team |
| Safe model loading, hash verification, strict key check (`kneeai/engine.py`) | Developed by the team |
| Run recording, reference-label comparison, reproducibility check, visual output, this report (`kneeai/cli.py`, `kneeai/report.py`, `kneeai/visualize.py`) | Developed by the team |
| Automated regression and end-to-end tests on synthetic data (`tests/`, GitHub Actions) | Developed by the team |

## 3. Environment

| Item | Value |
|---|---|
""" + "\n".join(f"| {k} | {v} |" for k, v in env.items()) + f"""
| Third-party model commit | `{PINNED_COMMIT}` |
| Checkpoint size | {model_check.get('size_bytes', 'n/a')} bytes |
| Checkpoint SHA-256 | `{model_check.get('sha256', 'n/a')}` |
| Parameters | {model_check.get('n_params', 'n/a')} |

## 4. Technical acceptance checks

| Check | Result |
|---|---|
{check_md}

## 5. Test data

- Source: clinical partner archive, exported as complete DICOM studies and pseudonymised on the partner's
  premises before transfer. Re-identification key retained by the partner.
- Number of studies: {ok['case_code'].nunique()} completed, {failed['case_code'].nunique() if not failed.empty else 0} failed.
- Case selection method: [TO FILL – who selected the cases, on what criteria, date]
- Reference labels: recorded by [radiologist 1] and [radiologist 2] independently of model output;
  disagreements resolved by [method]. Values: 1 = tear present, 0 = absent, U = uncertain.

## 6. Series used per study

{_md_table(series_tbl)}
## 7. Model outputs vs. radiologist reference (latest run)

{_md_table(results_tbl)}
Agreement on labelled targets: {n_agree} agree, {n_dis} disagree (of {n_agree + n_dis}).
These counts describe this small demonstration set only and are **not** performance estimates
(sensitivity/specificity cannot be established from a handful of cases).

## 8. Reproducibility

{_md_table(rep_summary)}
## 9. Failed runs

{_md_table(failed)}
## 10. Limitations and findings

- **Third-party checkpoint maturity.** The checkpoint's own metadata records `epoch = 1`, `best_epoch = 1`
  and `best_metric = 0.5` from a planned 25-epoch run. This indicates minimal training; scores from this
  checkpoint are not expected to carry reliable diagnostic information. The demonstration therefore
  validates the **pipeline**, not diagnostic accuracy. Model training/fine-tuning on partner data is
  project work, not completed work.
- Thresholds are the third-party defaults (ACL 0.05, others 0.50); they are not established for this data
  and were not tuned on these cases.
- PCL and LCL are not outputs of this model. Negative target predictions do not mean the knee is normal.
- Series selection uses a rule-based classifier of plane/sequence (v0.1); selections were reviewed by
  [radiographer] on [date]: [TO FILL – confirmed / corrections].
- Rights: the repository is MIT-licensed; whether the licence covers the checkpoint and what terms apply to
  its training data is [TO FILL – see docs/MODEL_RIGHTS.md].

## 11. Next validation steps

1. Obtain or train a model with documented training and validation results.
2. Expand to a larger, independently labelled partner dataset (planned within the project).
3. Calibrate thresholds on a separate tuning set; evaluate on a held-out test set.

## Annexes

- `outputs/results.csv` – all runs, all targets
- `outputs/<case>/selection.csv` – series inventory and selection per study
- `outputs/<case>/montage_<run>.png` – exact model input slices and scores (privacy-reviewed)
- `outputs/environment.json`, `outputs/model_check.json`, `outputs/run_log.txt`
"""
    out = os.path.join(outputs_dir, f"TECHNICAL_DEMONSTRATION_REPORT_{today}.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(md)
    return out

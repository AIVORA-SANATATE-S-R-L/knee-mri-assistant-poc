# Technical demonstration report supporting TRL assessment

**Project:** AI assistant for knee MRI (PoCIDIF 2021–2027, Action 2.1)
**Report date:** 2026-10-27
**Test period:** 2026-10-27T12:01:34+02:00 to 2026-10-27T12:01:42+02:00
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
| Neural network architecture and weights (`rsna-knee-mri-model`, commit `a254152c1e7c`) | Third-party, MIT-licensed repository. Not developed by the team. |
| Pseudonymisation and identifier check (`kneeai/deid.py`) | Developed by the team |
| Study indexing, sequence classification and explicit series selection (`kneeai/dicom_index.py`, `kneeai/series_select.py`) | Developed by the team |
| Safe model loading, hash verification, strict key check (`kneeai/engine.py`) | Developed by the team |
| Run recording, reference-label comparison, reproducibility check, visual output, this report (`kneeai/cli.py`, `kneeai/report.py`, `kneeai/visualize.py`) | Developed by the team |

## 3. Environment

| Item | Value |
|---|---|
| recorded_at | 2026-10-27T12:01:29+02:00 |
| os | Linux-6.18.44-fc-v165-x86_64-with-glibc2.39 |
| machine | x86_64 |
| processor | x86_64 |
| python | 3.13.17 |
| python_executable_arch | 64bit |
| torch | 2.14.1+cu130 |
| torchvision | 0.29.1+cu130 |
| cuda_available | False |
| gpu | none (CPU inference) |
| numpy | 2.5.3 |
| pandas | 3.0.5 |
| pydicom | 3.0.2 |
| opencv | 5.0.0 |
| project_commit | a34f18f2164a8b6b29ad66c14cb04e4edade97ea |
| ram_gb | 8.4 |
| Third-party model commit | `a254152c1e7cf4aa1d7391cb9260d54a4185118c` |
| Checkpoint size | 152040442 bytes |
| Checkpoint SHA-256 | `68bd5d3fd040fa6e4c464d99d2c7dcacaebd2a40388ec0902133d4e26be0f791` |
| Parameters | 12656716 |

## 4. Technical acceptance checks

| Check | Result |
|---|---|
| Full checkpoint downloaded (not an LFS pointer) | PASSED |
| Checkpoint SHA-256 matches pinned value | PASSED |
| Safe loading with weights_only=True | PASSED |
| Strict key match (no missing/unexpected keys) | PASSED |
| Finite outputs on all completed cases | PASSED |
| Repeated runs identical within tolerance 1e-4 | PASSED |
| No failed cases | NOT MET / PENDING |

## 5. Test data

- Source: clinical partner archive, exported as complete DICOM studies and pseudonymised on the partner's
  premises before transfer. Re-identification key retained by the partner.
- Number of studies: 1 completed, 1 failed.
- Case selection method: [TO FILL – who selected the cases, on what criteria, date]
- Reference labels: recorded by [radiologist 1] and [radiologist 2] independently of model output;
  disagreements resolved by [method]. Values: 1 = tear present, 0 = absent, U = uncertain.

## 6. Series used per study

| case_code | selected_series | n_series_used | total_seconds |
|---|---|---|---|
| SYN-001 | S1_SAG_FLUID_FS (PD_TSE_FS_SAG, 24 img); S2_AX_FLUID_FS (PD_TSE_FS_AX, 26 img); S3_COR_NONFLUID_NOFS (T1_TSE_COR, 22 img) | 3 | 4.14 |

## 7. Model outputs vs. radiologist reference (latest run)

| case_code | target | score | threshold | predicted_label | radiologist_reference | agreement |
|---|---|---|---|---|---|---|
| SYN-001 | ACL | 0.5337 | 0.05 | 1.0 | 0 | disagree |
| SYN-001 | MCL | 0.5102 | 0.5 | 1.0 | 0 | disagree |
| SYN-001 | Medial Meniscus | 0.5024 | 0.5 | 1.0 | U | reference uncertain |
| SYN-001 | Lateral Meniscus | 0.5136 | 0.5 | 1.0 | 0 | disagree |

Agreement on labelled targets: 0 agree, 3 disagree (of 3).
These counts describe this small demonstration set only and are **not** performance estimates
(sensitivity/specificity cannot be established from a handful of cases).

## 8. Reproducibility

| case_code | runs | max_difference |
|---|---|---|
| SYN-001 | 2 | 0.00e+00 |

## 9. Failed runs

| run_label | case_code | error |
|---|---|---|
| refuse_check | SYN-RAW | RuntimeError: study still contains identifiers (525 findings, e.g. PatientName); pseudonymise it at ELDA first (python -m kneeai deid) |

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
- `outputs/environment.json`, `outputs/model_check.json`

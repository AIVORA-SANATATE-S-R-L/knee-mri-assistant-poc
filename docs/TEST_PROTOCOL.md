# Test protocol – demonstration on clinical partner (ELDA) studies

Version 0.2 · prepared 2026-10-27

## 0. Before touching any patient data

- [ ] Written authorisation from ELDA (data controller) for this internal research test, stating purpose,
      number of studies, pseudonymisation, where data is processed and when it is deleted. Even a short signed
      letter/annex to the collaboration agreement is better than nothing.
- [ ] Computer used for processing has disk encryption on (Windows: BitLocker / Device encryption) and a
      password; no cloud sync (OneDrive/Google Drive) on the data folders.
- [ ] `verify-model` and `smoke-test` pass on that computer.

## 1. Case selection (radiologists)

- At least **2** complete studies: one with a confirmed target injury (ACL, MCL, medial or lateral meniscus
  tear) and one without any of these. Better: 6–10 studies covering each target and one or two difficult cases.
- Record how and by whom cases were selected (criteria, date).
- Each case gets a code `ELDA-K001`, `ELDA-K002`, …

## 2. Export (at ELDA / under ELDA authorisation)

- From the PACS, export each **full study** (all series, original DICOM, not screenshots, not single slices)
  into its own folder, e.g. `D:\ELDA_RAW\K001\`.
- If the PACS offers an anonymised export, use it **and** still run step 3.

## 3. Pseudonymise immediately

```powershell
.\.venv\Scripts\python -m kneeai deid --input D:\ELDA_RAW\K001 --output C:\KneeDemo\pseudo --case ELDA-K001 --key-file D:\ELDA_KEY\key.csv
```

The command asks for the secret salt (hidden). Use the same salt for every case; ELDA keeps it.

- The key file and the salt stay with ELDA, never in the repository, never sent to IT.
- Open a few images of every series and check for burned-in names/dates. If present, exclude that series.
- Delete the raw export (`D:\ELDA_RAW\K001`) once pseudonymisation is checked.

## 4. Reference labels (radiologists, before seeing any model output)

Copy `data/reference_labels_template.csv` to `data/reference_labels.csv` (not committed) and fill one row per
case: `1` = tear present, `0` = absent, `U` = uncertain. Two radiologists label independently; disagreements are
resolved and recorded.

## 5. Inspect series selection (radiographer)

```powershell
.\.venv\Scripts\python -m kneeai inspect --study C:\KneeDemo\pseudo\ELDA-K001
```

The radiographer confirms that the plane / fluid / fat-sat classification and the selected series are correct
(`outputs/ELDA-K001/selection.csv`). Any correction is recorded – do not change preprocessing to make scores look better.

## 6. Run, twice

```powershell
.\.venv\Scripts\python -m kneeai run --study C:\KneeDemo\pseudo\ELDA-K001 --case ELDA-K001 --run-label run1
.\.venv\Scripts\python -m kneeai run --study C:\KneeDemo\pseudo\ELDA-K001 --case ELDA-K001 --run-label run2
```

Failed runs are kept in `outputs/results.csv` and reported. No thresholds are tuned on these cases.

## 7. Report

```powershell
.\.venv\Scripts\python -m kneeai report
```

Fill the `[TO FILL]` fields, have it reviewed, sign and date it, export to PDF.

## Pass criteria (technical)

Complete input handling · safe successful loading · finite outputs · identical repeated runs (≤ 1e-4) ·
traceable versions (commit + checkpoint hash) · reviewed reference labels. Disagreements with the reference
are reported and investigated, not hidden. A handful of cases cannot establish sensitivity or specificity.

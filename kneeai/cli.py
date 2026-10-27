"""Command-line interface.  Usage:  python -m kneeai <command> [options]

Commands, in the order you normally use them:
  env           record software/hardware environment     -> outputs/environment.json
  verify-model  hash + safe load + dummy forward pass      -> outputs/model_check.json
  smoke-test    full pipeline on synthetic DICOMs (no patient data)
  deid          pseudonymise one study (run at ELDA)
  inspect       list series, identifiers left, selection   -> outputs/<case>/selection.csv
  run           run the model on one pseudonymised study   -> outputs/results.csv (+ json, png)
  report        build the dated technical report           -> outputs/TECHNICAL_DEMONSTRATION_REPORT_<date>.md
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
WORK = None  # None = temporary folder next to the study (same disk), always removed after use

RESULT_FIELDS = [
    "run_label", "run_timestamp", "case_code", "project_commit", "model_commit", "checkpoint_sha256",
    "selected_series", "n_series_used", "target", "score", "threshold", "predicted_label",
    "radiologist_reference", "agreement", "inference_seconds", "total_seconds", "error",
]


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _log(line: str) -> None:
    """Append-only execution log (no patient identifiers: only case codes)."""
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "run_log.txt"), "a", encoding="utf-8") as fh:
        fh.write(f"{_now()}  {line}\n")


def cmd_env(_a):
    from .envinfo import collect
    os.makedirs(OUT, exist_ok=True)
    info = collect(ROOT)
    with open(os.path.join(OUT, "environment.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    for k, v in info.items():
        print(f"{k:24} {v}")


def cmd_verify(_a):
    from .engine import ModelCheckError, SafeKneeEngine, PINNED_SHA256, PINNED_COMMIT
    os.makedirs(OUT, exist_ok=True)
    rec = {"checked_at": _now(), "model_commit": PINNED_COMMIT, "pinned_sha256": PINNED_SHA256}
    try:
        e = SafeKneeEngine()
        rec.update({
            "size_bytes": e.ckpt_size, "sha256": e.ckpt_sha256, "sha256_match": e.ckpt_sha256 == PINNED_SHA256,
            "safe_load_ok": True, "strict_keys_ok": not e.load_report["missing_keys"] and not e.load_report["unexpected_keys"],
            "load_report": e.load_report, "n_params": e.n_params, "device": str(e.device),
            "dummy_forward": e.dummy_check(),
        })
        import torch
        ck = torch.load(os.path.join(e.model_dir, "best_model.pth"), map_location="cpu", weights_only=True)
        rec["checkpoint_training_metadata"] = {k: ck.get(k) for k in ("epoch", "best_epoch", "best_metric")}
        rec["note"] = "Dummy-input result proves only that the software executes."
        status = "PASSED"
    except ModelCheckError as exc:
        rec.update({"safe_load_ok": False, "error": str(exc)})
        status = "FAILED"
    with open(os.path.join(OUT, "model_check.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2, default=str)
    print(json.dumps(rec, indent=2, default=str))
    print(f"\nMODEL CHECK: {status}")
    _log(f"verify-model status={status} sha256={rec.get('sha256', '')}")
    if status != "PASSED":
        sys.exit(1)


def _get_salt(a) -> str:
    """Secret salt: from --salt (tests only), env KNEEAI_SALT, or typed in hidden.
    Typing it keeps it out of the PowerShell history."""
    if a.salt:
        return a.salt
    if os.environ.get("KNEEAI_SALT"):
        return os.environ["KNEEAI_SALT"]
    import getpass
    s1 = getpass.getpass("Secret salt (kept by ELDA, >= 12 characters, same for every case): ")
    if not os.path.exists(a.key_file):
        s2 = getpass.getpass("Repeat secret salt: ")
        if s1 != s2:
            sys.exit("Salts do not match.")
    return s1


def cmd_deid(a):
    from .deid import find_identifiers, pseudonymise_study
    counts = pseudonymise_study(a.input, a.output, a.case, _get_salt(a), a.key_file)
    print(f"Pseudonymised {a.case}: {counts}")
    left = find_identifiers(os.path.join(a.output, a.case))
    if left:
        print(f"WARNING: {len(left)} identifier findings remain, review before transfer:")
        for f, tag, s in left[:30]:
            print("  ", f, tag, s)
    else:
        print("Header check (incl. nested sequences): no names, dates, free text, identifiers or private tags remain.")
    print("REMINDER: open the images and check visually for burned-in names/text before transfer.")


def cmd_inspect(a):
    from .deid import find_identifiers
    from .series_select import stage_study
    case = a.case or os.path.basename(a.study.rstrip("/\\"))
    out_dir = os.path.join(OUT, case)
    os.makedirs(out_dir, exist_ok=True)
    left = find_identifiers(a.study)
    staged, meta, table, chosen, errors = stage_study(a.study, case, WORK, copy_files=False)
    table.to_csv(os.path.join(out_dir, "selection.csv"), index=False)
    import pandas as pd
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.drop(columns=["series_uid"]).to_string(index=False))
    print(f"\nSelected {len(chosen)} series for the model.")
    if left:
        print(f"\nIDENTIFIER CHECK: {len(left)} findings, the study is NOT pseudonymised. Do not process it outside ELDA.")
        for f, tag, s in left[:15]:
            print("  ", tag, s, "-", os.path.relpath(f, a.study))
    else:
        print("IDENTIFIER CHECK: no listed identifiers found in headers.")
    for e in errors:
        print("ERROR:", e)


def _load_reference(path):
    import pandas as pd
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    return {r["case_code"]: r for _, r in df.iterrows()}


def _agreement(pred: int, ref: str) -> str:
    ref = (ref or "").strip().upper()
    if ref in ("", "NA"):
        return "no reference"
    if ref == "U":
        return "reference uncertain"
    try:
        return "agree" if int(float(ref)) == pred else "disagree"
    except ValueError:
        return "no reference"


def cmd_run(a):
    from .deid import find_identifiers
    from .engine import PINNED_COMMIT, TARGETS_REPORTED, SafeKneeEngine
    from .envinfo import _git_head
    from .series_select import stage_study
    from .visualize import save_montage

    case = a.case
    run_label = a.run_label or dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT, case)
    os.makedirs(out_dir, exist_ok=True)
    refs = _load_reference(a.labels)
    targets = TARGETS_REPORTED if not a.all_targets else None
    t_start = time.time()
    row_base = {"run_label": run_label, "run_timestamp": _now(), "case_code": case,
                "project_commit": _git_head(ROOT), "model_commit": PINNED_COMMIT}
    rows = []
    staged = None
    res_path = os.path.join(OUT, "results.csv")
    if os.path.exists(res_path):
        import pandas as pd
        prev = pd.read_csv(res_path, usecols=["run_label", "case_code"], dtype=str)
        if ((prev["run_label"] == run_label) & (prev["case_code"] == case)).any():
            sys.exit(f"Run label '{run_label}' already used for {case}; choose another (e.g. run3).")
    try:
        if not a.allow_identifiers:
            left = find_identifiers(a.study)
            if left:
                raise RuntimeError(f"study still contains identifiers ({len(left)} findings, e.g. {left[0][1]}); "
                                   "pseudonymise it at ELDA first (python -m kneeai deid)")
        engine = SafeKneeEngine()
        staged, meta, table, chosen, errors = stage_study(a.study, case, WORK)
        table.to_csv(os.path.join(out_dir, "selection.csv"), index=False)
        if errors:
            raise RuntimeError("; ".join(errors))
        res = engine.predict_staged(staged, meta)
        targets = targets or engine.targets
        labels = [f"S{k}: {s.description or s.short_label()}" for k, s in enumerate(chosen, 1)]
        save_montage(res["images_tensor"], labels, res["probabilities"], res["thresholds"], case,
                     os.path.join(out_dir, f"montage_{run_label}.png"), targets)
        detail = {k: v for k, v in res.items() if k != "images_tensor"}
        detail.update({"case_code": case, "run_label": run_label, "checkpoint_sha256": engine.ckpt_sha256,
                       "selected_series": meta.to_dict(orient="records")})
        with open(os.path.join(out_dir, f"result_{run_label}.json"), "w", encoding="utf-8") as fh:
            json.dump(detail, fh, indent=2)
        sel = "; ".join(f"{r['SeriesInstanceUID']} ({r['source_description']}, {r['n_images']} img)" for _, r in meta.iterrows())
        ref_row = refs.get(case, {})
        for t in targets:
            ref = ref_row.get(t, "") if len(ref_row) else ""
            pred = res["binary"][t]
            rows.append({**row_base, "checkpoint_sha256": engine.ckpt_sha256, "selected_series": sel,
                         "n_series_used": len(chosen), "target": t, "score": round(res["probabilities"][t], 6),
                         "threshold": res["thresholds"][t], "predicted_label": pred, "radiologist_reference": ref,
                         "agreement": _agreement(pred, ref), "inference_seconds": res["inference_seconds"],
                         "total_seconds": round(time.time() - t_start, 2), "error": ""})
        print(f"{case} [{run_label}] OK in {time.time() - t_start:.1f}s")
        for r in rows:
            print(f"  {r['target']:<17} score={r['score']:.4f} thr={r['threshold']:.2f} pred={r['predicted_label']} "
                  f"ref={r['radiologist_reference'] or '-'} -> {r['agreement']}")
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        rows.append({**row_base, "checkpoint_sha256": "", "selected_series": "", "n_series_used": 0, "target": "",
                     "score": "", "threshold": "", "predicted_label": "", "radiologist_reference": "",
                     "agreement": "", "inference_seconds": "", "total_seconds": round(time.time() - t_start, 2),
                     "error": msg})
        print(f"{case} [{run_label}] FAILED: {msg}")
        with open(os.path.join(out_dir, f"error_{run_label}.log"), "w", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
    finally:
        if staged:
            shutil.rmtree(os.path.dirname(staged), ignore_errors=True)  # never leave copies of studies behind
    _log(f"run case={case} label={run_label} commit={row_base['project_commit']} "
         f"status={'FAILED ' + rows[-1]['error'] if rows and rows[-1]['error'] else 'OK'}")
    new = not os.path.exists(res_path)
    with open(res_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def cmd_report(_a):
    from .report import build_report
    print("Report written:", build_report(OUT))


def cmd_smoke(_a):
    from tests.test_smoke import run_smoke
    run_smoke()


def main(argv=None):
    p = argparse.ArgumentParser(prog="kneeai", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env").set_defaults(func=cmd_env)
    sub.add_parser("verify-model").set_defaults(func=cmd_verify)
    sub.add_parser("smoke-test").set_defaults(func=cmd_smoke)

    d = sub.add_parser("deid", help="pseudonymise one study (run at ELDA)")
    d.add_argument("--input", required=True, help="folder with the original exported study")
    d.add_argument("--output", required=True, help="root folder for pseudonymised studies")
    d.add_argument("--case", required=True, help="case code, e.g. ELDA-K001")
    d.add_argument("--salt", help=argparse.SUPPRESS)  # tests only; normally typed in hidden or KNEEAI_SALT
    d.add_argument("--key-file", required=True, help="CSV mapping case code -> original UID (stays at ELDA)")
    d.set_defaults(func=cmd_deid)

    i = sub.add_parser("inspect", help="list series, check identifiers, show selection")
    i.add_argument("--study", required=True)
    i.add_argument("--case")
    i.set_defaults(func=cmd_inspect)

    r = sub.add_parser("run", help="run the model on one pseudonymised study")
    r.add_argument("--study", required=True)
    r.add_argument("--case", required=True)
    r.add_argument("--labels", default=os.path.join(ROOT, "data", "reference_labels.csv"))
    r.add_argument("--run-label", help="e.g. run1, run2 (repeat with the same environment)")
    r.add_argument("--all-targets", action="store_true", help="report all 12 outputs, not just the 4 targets")
    r.add_argument("--allow-identifiers", action="store_true", help=argparse.SUPPRESS)
    r.set_defaults(func=cmd_run)

    sub.add_parser("report").set_defaults(func=cmd_report)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()

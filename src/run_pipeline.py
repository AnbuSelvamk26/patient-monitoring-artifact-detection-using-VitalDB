"""End-to-end run: load -> clean -> detect -> evaluate -> save results.

    python src/run_pipeline.py                 # real VitalDB data (needs internet, pip install vitaldb)
    python src/run_pipeline.py --n-cases 60    # more patients
    python src/run_pipeline.py --demo          # simulated data, only to test that the code runs
"""
import argparse
import json
import os

import pandas as pd

from preprocess import load_vitaldb, clean
from detect import detect
from evaluate import run_evaluation
from plots import plot_case, plot_class_mix, plot_confusion, plot_sensitivity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=float, default=4.0)
    ap.add_argument("--demo", action="store_true", help="use simulated data (NOT for reporting results)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    a = ap.parse_args()
    out = os.path.abspath(a.out); figs = os.path.join(out, "figures"); os.makedirs(figs, exist_ok=True)

    if a.demo:
        from synthetic import make_demo_data
        raw = make_demo_data(12, a.seed)
    else:
        raw = load_vitaldb(a.n_cases, a.seed)
    df, n_imp = clean(raw)
    print(f"Rows: {len(raw):,}  impossible values removed: {n_imp:,}")

    # 1) detector on the real (unlabeled) recordings
    flags, ep, _ = detect(df, K=a.K)
    ep.to_csv(os.path.join(out, "episodes.csv"), index=False)
    mix = ep.label.value_counts()
    hours = len(df) * 2 / 3600
    summary = {"data": "SIMULATED (demo)" if a.demo else "VitalDB", "cases": int(df.caseid.nunique()),
               "rows": int(len(df)), "hours": round(hours, 1), "impossible_values_removed": n_imp, "K": a.K,
               "episodes_total": int(len(ep)), "episodes_per_hour": round(len(ep) / hours, 2),
               "episodes_by_class": {k: int(v) for k, v in mix.items()},
               "share_artifact_pct": round(100 * mix.get("artifact", 0) / max(len(ep), 1), 1),
               "share_real_event_pct": round(100 * mix.get("real_event", 0) / max(len(ep), 1), 1),
               "share_review_pct": round(100 * mix.get("review", 0) / max(len(ep), 1), 1)}
    best = int(df.dropna(subset=["HR", "SpO2", "MAP"]).groupby("caseid").size().idxmax())
    plot_case(df, ep, best, os.path.join(figs, "detected_episodes_case.png"))
    plot_class_mix(ep, os.path.join(figs, "episode_mix_by_vital.png"))
    if "department" in df:
        dept = ep.caseid.map(df.groupby("caseid").department.first())
        ep.assign(department=dept).groupby(["department", "label"]).size().unstack(fill_value=0) \
            .to_csv(os.path.join(out, "episodes_by_department.csv"))

    # 2) evaluation with injected ground truth
    ev, _, _, _ = run_evaluation(df, K=a.K, seed=0)
    conf = ev.pop("confusion"); sens = pd.DataFrame(ev.pop("sensitivity"))
    conf.to_csv(os.path.join(out, "confusion_matrix.csv")); sens.to_csv(os.path.join(out, "sensitivity_K.csv"), index=False)
    plot_confusion(conf, os.path.join(figs, "confusion_matrix.png")); plot_sensitivity(sens, os.path.join(figs, "sensitivity_K.png"))
    summary["evaluation"] = ev
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(summary, indent=2, default=str)); print(conf); print(sens)


if __name__ == "__main__":
    main()

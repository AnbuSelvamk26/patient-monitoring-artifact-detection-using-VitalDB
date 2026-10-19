"""Evaluation of the detector.

VitalDB has no labels for "artifact" or "real event", so the detector is tested with
*injected* ground truth: known artifacts (short single-vital spikes) and known physiological
events (several vitals moving together for 1-3 minutes) are added to real recordings.
The test patients are separate from the patients used to fit the Isolation-Forest baseline.

Caveat: naturally occurring anomalies in the recordings are unlabeled, so they count as false
alarms in the row-level precision of EVERY method. Use the numbers to compare methods and
settings with each other, not as absolute clinical accuracy.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from preprocess import VITALS, PLAUSIBLE, ABNORMAL
from detect import ARTIFACT, REAL, REVIEW, compute_residuals, make_flags, extract_episodes

ART_RANGE = {"HR": (20, 70), "SpO2": (8, 25), "MAP": (25, 70), "Temp": (1.0, 3.0), "RR": (8, 20)}
EVENT_TYPES = {
    "hypotension":  {"MAP": (-40, -20), "HR": (12, 30)},
    "hypoxaemia":   {"SpO2": (-15, -7), "HR": (10, 20), "RR": (5, 10)},
    "hypertension": {"MAP": (30, 50), "HR": (15, 30)},
}


def inject(df, rng, n_art=15, n_evt=8):
    """Return (df_injected, truth[n x V int8: 0 none, 1 artifact, 2 event], list of injections)."""
    d = df.copy()
    V = len(VITALS)
    X = d[VITALS].to_numpy(float).copy()
    truth = np.zeros((len(d), V), dtype=np.int8)
    inj = []
    for cid, pos in d.groupby("caseid").indices.items():
        n = len(pos)
        if n < 1200:
            continue
        occ = np.zeros(n, bool)
        jobs = ["art"] * n_art + ["evt"] * n_evt
        rng.shuffle(jobs)
        for kind in jobs:
            for _ in range(60):
                if kind == "art":
                    j = int(rng.integers(V)); span = int(rng.integers(1, 7)); vs = [j]
                else:
                    typ = str(rng.choice(list(EVENT_TYPES))); plateau = int(rng.integers(30, 91))
                    span = plateau + 10; vs = [VITALS.index(v) for v in EVENT_TYPES[typ]]
                s = int(rng.integers(200, n - span - 200))
                lo, hi = s - 200, s + span + 200
                if occ[lo:hi].any() or np.isnan(X[pos[lo:hi]][:, vs]).any():
                    continue
                if kind == "art":
                    v = VITALS[j]; delta = rng.uniform(*ART_RANGE[v]) * (-1 if v == "SpO2" else rng.choice([-1, 1]))
                    X[pos[s:s + span], j] += delta
                    truth[pos[s:s + span], j] = 1
                    rec = {"kind": ARTIFACT, "vitals": [v], "type": "spike"}
                else:
                    shape = np.concatenate([np.linspace(0, 1, 5), np.ones(plateau), np.linspace(1, 0, 5)])
                    for v, rg in EVENT_TYPES[typ].items():
                        jj = VITALS.index(v); X[pos[s:s + span], jj] += rng.uniform(*rg) * shape
                        truth[pos[s:s + span], jj] = 2
                    rec = {"kind": REAL, "vitals": list(EVENT_TYPES[typ]), "type": typ}
                for jj in vs:                                       # stay inside physically possible values
                    v = VITALS[jj]; X[pos[s:s + span], jj] = np.clip(X[pos[s:s + span], jj], *PLAUSIBLE[v])
                occ[lo:hi] = True
                inj.append({"caseid": cid, "start_i": s, "end_i": s + span - 1, **rec})
                break
    d[VITALS] = X
    return d, truth, inj


def match_episodes(inj, ep):
    """Predicted class of every injection ('missed' if nothing overlaps) + the unmatched (background) episodes."""
    by_case = {c: g for c, g in ep.groupby("caseid")}
    used, preds = set(), []
    for I in inj:
        g = by_case.get(I["caseid"]); pred = "missed"
        if g is not None:
            m = g[g["vital"].isin(I["vitals"]) & (g["start_i"] <= I["end_i"]) & (g["end_i"] >= I["start_i"])]
            if len(m):
                used.update(m.index); labs = set(m["label"])
                pred = REAL if REAL in labs else (ARTIFACT if ARTIFACT in labs else REVIEW)
        preds.append(pred)
    return preds, ep.loc[~ep.index.isin(used)]


def prf(truth, pred):
    tp = int((truth & pred).sum()); fp = int((~truth & pred).sum()); fn = int((truth & ~pred).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(p + r, 1e-9), 3)}


def _features(df, resid, med):
    return np.hstack([df[VITALS].fillna(med).to_numpy(float), resid.fillna(0).to_numpy(float)])


def run_evaluation(df_ts, K=4.0, seed=0, K_grid=(3, 4, 5, 6, 8)):
    rng = np.random.default_rng(seed)
    cases = np.array(sorted(df_ts.caseid.unique())); rng.shuffle(cases)
    n_test = max(2, int(round(0.3 * len(cases))))
    test_cases, train_cases = set(cases[:n_test]), set(cases[n_test:])
    out = {"n_train_cases": len(train_cases), "n_test_cases": len(test_cases), "K": K}

    dtest = df_ts[df_ts.caseid.isin(test_cases)].reset_index(drop=True)
    dtrain = df_ts[df_ts.caseid.isin(train_cases)].reset_index(drop=True)
    dtest_i, truth, inj = inject(dtest, rng)
    dtrain_i, _, _ = inject(dtrain, rng)
    hours = len(dtest_i) * 2 / 3600
    truth_row = (truth > 0).any(axis=1)
    out["n_injected_artifacts"] = sum(i["kind"] == ARTIFACT for i in inj)
    out["n_injected_events"] = sum(i["kind"] == REAL for i in inj)
    out["test_hours"] = round(hours, 1)

    resid, scale = compute_residuals(dtest_i)

    # ---- our detector at several K (sensitivity) -------------------------------------------------
    sens, main = [], None
    for k in sorted(set(K_grid) | {K}):
        flags = make_flags(resid, scale, k)
        ep = extract_episodes(dtest_i, resid, flags)
        preds, bg = match_episodes(inj, ep)
        kinds = [i["kind"] for i in inj]
        det = np.mean([p != "missed" for p in preds])
        correct = np.mean([p == kd for p, kd in zip(preds, kinds)])
        row = prf(truth_row, flags.any(axis=1).to_numpy())
        sens.append({"K": k, "detection_recall": round(det, 3), "correct_class_rate": round(correct, 3),
                     "background_episodes_per_hour": round(len(bg) / hours, 1), "row_f1": row["f1"]})
        if k == K:
            main = (flags, ep, preds, kinds, row)
    out["sensitivity"] = sens
    flags, ep, preds, kinds, row = main
    classes = [ARTIFACT, REAL, REVIEW, "missed"]
    conf = pd.DataFrame([[sum(1 for p, kd in zip(preds, kinds) if kd == t and p == c) for c in classes]
                         for t in (ARTIFACT, REAL)], index=["true_artifact", "true_real_event"],
                        columns=["pred_artifact", "pred_real_event", "pred_review", "missed"])
    out["confusion"] = conf
    out["ours_row_level"] = row
    for t, name in ((ARTIFACT, "artifact"), (REAL, "real_event")):
        idx = [i for i, kd in enumerate(kinds) if kd == t]
        out[f"{name}_detected"] = round(float(np.mean([preds[i] != "missed" for i in idx])), 3)
        out[f"{name}_classified_correctly"] = round(float(np.mean([preds[i] == t for i in idx])), 3)

    # ---- baselines (row level, same test rows) ----------------------------------------------------
    clin = np.zeros(len(dtest_i), bool)
    for v, (lo, hi) in ABNORMAL.items():
        clin |= ((dtest_i[v] < lo) | (dtest_i[v] > hi)).to_numpy()
    out["baseline_clinical_threshold"] = prf(truth_row, clin)

    resid_tr, scale_tr = compute_residuals(dtrain_i)
    med = dtrain_i[VITALS].median()
    rate = float(make_flags(resid_tr, scale_tr, K).any(axis=1).mean())       # same alert burden as ours
    iso = IsolationForest(n_estimators=100, contamination=min(max(rate, 0.001), 0.5), random_state=seed, n_jobs=-1)
    iso.fit(_features(dtrain_i, resid_tr, med))
    pred_if = iso.predict(_features(dtest_i, resid, med)) == -1
    out["baseline_isolation_forest"] = prf(truth_row, pred_if)
    return out, ep, dtest_i, inj

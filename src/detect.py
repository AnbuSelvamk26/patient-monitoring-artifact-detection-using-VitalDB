"""Artifact-vs-real-event detector (rule based, no training needed).

Idea
----
1. Every vital is compared with the patient's own recent baseline (median of the
   previous 5 minutes). A reading is *flagged* when it departs from that baseline by more
   than K x a robust, patient-specific scale.
2. Consecutive flagged samples of one vital form an *episode*.
3. Each episode is classified by looking at the other vitals around it:
     - other vitals also flagged within +-30 s  -> "real_event"  (several signals move together)
     - nobody else flagged and duration <= 30 s -> "artifact"    (isolated spike on one sensor)
     - otherwise                                -> "review"      (single-vital but sustained: unclear)
"""
import numpy as np
import pandas as pd

from preprocess import VITALS, INTERVAL

ARTIFACT, REAL, REVIEW = "artifact", "real_event", "review"
FLOOR = {"HR": 3.0, "SpO2": 1.5, "MAP": 5.0, "Temp": 0.2, "RR": 1.5}   # minimum scale per vital (units of the vital)


def compute_residuals(df, win=150, min_periods=30, vitals=VITALS):
    """Deviation from the patient's trailing baseline (median of previous `win` samples = 5 min)
    and a robust per-patient scale. df must have a RangeIndex and be sorted by caseid, time."""
    resid = pd.DataFrame(index=df.index, columns=vitals, dtype=float)
    scale = pd.DataFrame(index=df.index, columns=vitals, dtype=float)
    for v in vitals:
        base = df.groupby("caseid")[v].transform(lambda s: s.shift(1).rolling(win, min_periods=min_periods).median())
        r = df[v] - base
        mad = r.abs().groupby(df["caseid"]).transform("median").fillna(0)
        resid[v] = r
        scale[v] = np.maximum(FLOOR[v], 1.4826 * mad)
    return resid, scale


def make_flags(resid, scale, K=4.0):
    """Boolean table: True where |deviation| > K x scale."""
    return (resid.abs() > K * scale).fillna(False)


def extract_episodes(df, resid, flags, max_gap=2, co_win=15, max_artifact_s=30):
    """Group flagged samples into episodes and classify them (see module docstring)."""
    vit = list(flags.columns)
    F_all = flags.to_numpy()
    R_all = resid.to_numpy(float)
    t_all = df["time_s"].to_numpy()
    rows = []
    for cid, pos in df.groupby("caseid").indices.items():
        F, R, t = F_all[pos], R_all[pos], t_all[pos]
        n = len(F)
        cs = np.vstack([np.zeros((1, F.shape[1]), dtype=int), np.cumsum(F, axis=0)])      # prefix sums
        for j, v in enumerate(vit):
            p = np.flatnonzero(F[:, j])
            if p.size == 0:
                continue
            for g in np.split(p, np.flatnonzero(np.diff(p) > max_gap + 1) + 1):
                s, e = int(g[0]), int(g[-1])
                lo, hi = max(0, s - co_win), min(n - 1, e + co_win)
                cnt = cs[hi + 1] - cs[lo]
                others = [vit[k] for k in range(len(vit)) if k != j and cnt[k] > 0]
                dur = (e - s + 1) * INTERVAL
                if others:
                    label = REAL
                elif dur <= max_artifact_s:
                    label = ARTIFACT
                else:
                    label = REVIEW
                seg = R[s:e + 1, j]
                peak = float(seg[np.nanargmax(np.abs(seg))]) if np.isfinite(seg).any() else np.nan
                rows.append({"caseid": cid, "vital": v, "start_i": s, "end_i": e, "start_s": int(t[s]),
                             "end_s": int(t[e]), "duration_s": dur, "peak_dev": round(peak, 2),
                             "n_other": len(others), "others": ",".join(others), "label": label})
    cols = ["caseid", "vital", "start_i", "end_i", "start_s", "end_s", "duration_s", "peak_dev",
            "n_other", "others", "label"]
    return pd.DataFrame(rows, columns=cols)


def detect(df, K=4.0, **kw):
    """Convenience wrapper: returns (flags, episodes, resid)."""
    resid, scale = compute_residuals(df)
    flags = make_flags(resid, scale, K)
    return flags, extract_episodes(df, resid, flags, **kw), resid

"""Loading and cleaning of the VitalDB sample.

The cleaning rules are exactly the ones used in notebooks/01_EDA_VitalDB.ipynb:
impossible values -> NaN, temperature probe warm-up -> NaN, short gaps bridged
by carrying the last reading forward inside each patient.
"""
import numpy as np
import pandas as pd

VITALS = ["HR", "SpO2", "MAP", "Temp", "RR"]
TRACKS = {"HR": "Solar8000/HR", "SpO2": "Solar8000/PLETH_SPO2", "MAP": "Solar8000/ART_MBP",
          "Temp": "Solar8000/BT", "RR": "Solar8000/RR_CO2"}
UNITS = {"HR": "beats/min", "SpO2": "%", "MAP": "mmHg", "Temp": "°C", "RR": "breaths/min"}
INTERVAL = 2                      # seconds between samples
WARMUP_S = 600                    # temperature probe warm-up (first 10 minutes)
PLAUSIBLE = {"HR": (20, 250), "SpO2": (50, 100), "MAP": (20, 250), "Temp": (25, 45), "RR": (1, 60)}
ABNORMAL = {"HR": (50, 100), "SpO2": (92, np.inf), "MAP": (60, 120), "Temp": (35, 38), "RR": (8, 20)}
GAP_S = {"HR": 10, "SpO2": 10, "MAP": 10, "Temp": 120, "RR": 20}   # seconds a reading may be carried forward


def load_vitaldb(n_cases=30, seed=42):
    """Download n_cases random surgical cases that have all five vitals (needs internet + `pip install vitaldb`)."""
    import vitaldb

    tnames = ",".join(TRACKS.values())
    caseids = vitaldb.find_cases(tnames)
    chosen = sorted(np.random.default_rng(seed).choice(caseids, size=min(n_cases, len(caseids)), replace=False))
    frames = []
    for cid in chosen:
        arr = vitaldb.load_case(int(cid), list(TRACKS.values()), INTERVAL)
        d = pd.DataFrame(arr, columns=VITALS)
        d.insert(0, "time_s", np.arange(len(d)) * INTERVAL)
        d.insert(0, "caseid", int(cid))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    try:
        try:
            clin = pd.read_csv("https://api.vitaldb.net/cases")
        except Exception:
            clin = vitaldb.load_clinical_data()
        keep = [c for c in ["caseid", "age", "sex", "department", "asa"] if c in clin.columns]
        df = df.merge(clin[keep], on="caseid", how="left")
    except Exception as e:                                   # clinical table is optional
        print("Clinical table not available, continuing without it:", e)
    return df


def clean(df_raw):
    """Return (df_ts, n_impossible). df_ts keeps gaps that could not be bridged (time-series friendly)."""
    df = df_raw.copy()
    n_impossible = 0
    for v, (lo, hi) in PLAUSIBLE.items():
        bad = (df[v] < lo) | (df[v] > hi)
        n_impossible += int(bad.sum())
        df.loc[bad, v] = np.nan
    df.loc[df["time_s"] < WARMUP_S, "Temp"] = np.nan
    for v in VITALS:
        limit = max(1, GAP_S[v] // INTERVAL)
        df[v] = df.groupby("caseid")[v].transform(lambda s: s.ffill(limit=limit))
    return df.reset_index(drop=True), n_impossible

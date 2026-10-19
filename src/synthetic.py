"""Simulated monitor data, ONLY for testing the code without internet access (--demo).
Results obtained on it say nothing about real patients and must never be reported as results."""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from preprocess import VITALS, INTERVAL


def _ar1(n, phi, sd, rng):
    return lfilter([1.0], [1.0, -phi], rng.normal(0, sd, n))


def make_demo_data(n_cases=12, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(n_cases):
        n = int(rng.integers(6000, 9000))
        hr = 75 + 10 * rng.standard_normal() + _ar1(n, 0.995, 0.8, rng)
        mapp = 82 + 4 * rng.standard_normal() + _ar1(n, 0.995, 1.2, rng)
        spo2 = np.clip(100 - np.abs(_ar1(n, 0.9, 0.15, rng)), 94, 100).round()
        temp = 36.0 + _ar1(n, 0.9995, 0.01, rng)
        rr = (13 + rng.integers(-2, 3) + np.cumsum(rng.random(n) < 0.002) % 3).astype(float)
        d = pd.DataFrame({"caseid": 1000 + k, "time_s": np.arange(n) * INTERVAL,
                          "HR": hr.round(), "SpO2": spo2, "MAP": mapp.round(), "Temp": temp.round(1), "RR": rr})
        d.loc[:300, "Temp"] = np.nan                                   # probe not attached yet
        for v in VITALS:                                                # random short dropouts
            for _ in range(int(rng.integers(3, 10))):
                s = int(rng.integers(0, n - 30)); d.loc[s:s + int(rng.integers(2, 25)), v] = np.nan
        s = int(rng.integers(500, n - 500)); d.loc[s:s + 40, "MAP"] = 0.0    # arterial line flush (impossible value)
        d["age"] = float(rng.integers(30, 80)); d["sex"] = rng.choice(["M", "F"])
        d["department"] = rng.choice(["General surgery", "Thoracic surgery", "Urology", "Gynecology"])
        d["asa"] = float(rng.integers(1, 4))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)

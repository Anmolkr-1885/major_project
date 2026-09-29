"""
Creates demo_patients_SYNTHETIC.csv - a FAKE, randomly generated patient table
with OASIS-like columns. It exists only so the pipeline can be tested before the
real dataset is plugged in.  Never report results obtained on this file.

    python tools/make_demo_data.py --n 400 --out demo_patients_SYNTHETIC.csv
"""
import argparse

import numpy as np
import pandas as pd


def make(n=400, seed=7):
    rng = np.random.default_rng(seed)
    dem = rng.random(n) < 0.4
    age = np.where(dem, rng.normal(78, 7, n), rng.normal(72, 8, n)).clip(60, 96).round()
    mmse = np.where(dem, rng.normal(23, 4, n), rng.normal(28.4, 1.4, n)).clip(4, 30).round()
    educ = np.where(dem, rng.normal(13, 3, n), rng.normal(14.5, 3, n)).clip(6, 22).round()
    etiv = rng.normal(1480, 170, n).clip(1100, 2000).round()
    nwbv = (np.where(dem, 0.715, 0.752) - 0.0022 * (age - 75) + rng.normal(0, 0.028, n)).round(3)
    apoe4 = np.where(rng.random(n) < np.where(dem, 0.55, 0.25), "Yes", "No")
    df = pd.DataFrame({
        "PatientID": [f"P{i:04d}" for i in range(1, n + 1)],
        "Gender": np.where(rng.random(n) < 0.58, "F", "M"),
        "Age": age.astype(int),
        "EducationYears": educ.astype(int),
        "SES": rng.integers(1, 6, n).astype(float),
        "MMSE": mmse,
        "eTIV": etiv.astype(int),
        "nWBV": nwbv,
        "ASF": (1755 / etiv).round(3),
        "APOE4_carrier": apoe4,
        "Hypertension": (rng.random(n) < np.where(dem, 0.45, 0.35)).astype(int),
        "Diabetes": (rng.random(n) < np.where(dem, 0.20, 0.15)).astype(int),
        "Diagnosis": np.where(dem, "Demented", "Nondemented"),
    })
    df.loc[rng.random(n) < 0.08, "SES"] = np.nan          # realistic gaps
    df.loc[rng.random(n) < 0.03, "MMSE"] = np.nan
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="demo_patients_SYNTHETIC.csv")
    a = ap.parse_args()
    make(a.n, a.seed).to_csv(a.out, index=False)
    print("written", a.out)

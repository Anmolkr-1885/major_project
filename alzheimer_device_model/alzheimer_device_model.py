#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Device-aware (neuromorphic) Alzheimer's disease prediction
===========================================================
IGZO electrolyte-gated synaptic transistor, ZrO2 + MgO dual electrolyte.

HOW MANY MODELS?
----------------
One model per device stack  ->  3 device models (created automatically, one for
every `case` found in device_data/sensitization_index.csv):

    ZrO2_1_MgO_1              ZrO2 1.0 um / MgO 1.0 um
    ZrO2_1point5_MgO_point5   ZrO2 1.5 um / MgO 0.5 um
    ZrO2_point5_MgO_1point5   ZrO2 0.5 um / MgO 1.5 um

Each device model uses ALL the data of its own device:
  * the 8 measured sensitization curves (3 stimulus conditions 1st=2nd, 1st>2nd,
    1st<2nd  x  3 swept variables interval / pulse width / gate voltage; the
    1st=2nd gate-voltage sweep was not measured) -> 8 "virtual nodes" per input
    feature of a physical reservoir (this layer is NOT trained);
  * its transfer curve (Id-Vg at the read Vds) -> the conductance states in which
    the trained readout weights are stored.
The three stimulus conditions are therefore NOT separate models; they are the
node types inside every model (`--ablation` measures what each one contributes).
Two software models (logistic regression, MLP) are trained on the same folds only
as references, so the cost/benefit of the hardware can be reported.

ARCHITECTURE (per device case)
------------------------------
 patient CSV --impute / one-hot / min-max--> x in [0,1]^d
   --paired-pulse encoding (x sets interval | pulse width | amplitude)-->
   measured device response SI = A2/A1 (shape-preserving PCHIP interpolation)
   --> d*8 node states --per-node gain/offset--> linear readout (logistic regression)
   --> weights programmed as differential pairs (G+ - G-) using n_levels
       conductance states of the device (Vg window at read Vds) + optional noise
   --> P(class)

USAGE
-----
  # train + 5-fold cross-validation + figures + saved models
  python alzheimer_device_model.py train --data patients.csv --target Diagnosis

  # OASIS style: binarise CDR (0 = healthy, >0 = dementia), CDR itself is dropped
  python alzheimer_device_model.py train --data oasis_cross-sectional.csv \
         --target CDR --positive 0.5,1,2

  # predict new patients with a saved device model
  python alzheimer_device_model.py predict --model results/models/ZrO2_1_MgO_1.joblib \
         --data new_patients.csv --out predictions.csv

Run `python alzheimer_device_model.py train -h` for every option.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler

try:
    import joblib
except ImportError:  # joblib ships with scikit-learn, this is just a safety net
    from sklearn.externals import joblib  # type: ignore

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ----------------------------------------------------------------------------
# Default settings (all can be changed from the command line)
# ----------------------------------------------------------------------------
DEFAULTS = {
    "device_dir": "device_data",
    "read_vds": 1.0,          # V   read bias of the synapse (transfer curve used)
    "vg_min": -1.0,           # V   programmable gate-voltage window ...
    "vg_max": 3.0,            # V   ... (same window as the linear Id-Vg plot)
    "n_levels": 32,           #     conductance states per synapse device (5 bit)
    "noise": 0.0,             #     relative conductance variation (0.05 = 5 %)
    "folds": 5,
    "seed": 42,
    "Cs": "0.01,0.1,1,10",    #     inverse L2 strength grid of the readout
    "noise_grid": "0,0.02,0.05,0.1,0.2",
    "levels_grid": "4,8,16,32,64,128",
    "noise_repeats": 10,
    "max_missing": 0.5,       #     drop feature columns with more missing values
    "max_categories": 20,     #     text columns with more categories are dropped
}
CONDITIONS = ["equal", "first_greater", "first_less"]
CONDITION_LABEL = {"equal": "1st = 2nd", "first_greater": "1st > 2nd", "first_less": "1st < 2nd"}
VARIABLES = ["interval_ms", "pulse_width_ms", "gate_voltage_V"]
VARIABLE_LABEL = {"interval_ms": "interval (ms)", "pulse_width_ms": "pulse width (ms)",
                  "gate_voltage_V": "gate voltage (V)"}


# ============================================================================
# 1. Device library (reads the device CSV files)
# ============================================================================
@dataclass
class DeviceCase:
    name: str
    zro2_um: float
    mgo_um: float
    curves: dict            # (condition, variable) -> (x, SI) measured points
    vg: np.ndarray          # transfer curve at read Vds
    id_a: np.ndarray
    vds: float

    @property
    def label(self) -> str:
        if np.isfinite(self.zro2_um) and np.isfinite(self.mgo_um):
            return f"ZrO\u2082 {self.zro2_um:g} \u00b5m / MgO {self.mgo_um:g} \u00b5m"
        return self.name

    @property
    def label_ascii(self) -> str:
        return self.label.replace("\u2082", "2").replace("\u00b5", "u")

    def node_keys(self, conditions=None):
        def order(k):
            c = CONDITIONS.index(k[0]) if k[0] in CONDITIONS else 99
            v = VARIABLES.index(k[1]) if k[1] in VARIABLES else 99
            return (c, v, k)
        keys = sorted(self.curves, key=order)
        if conditions:
            keys = [k for k in keys if k[0] in conditions]
        return keys

    def reservoir_curves(self, conditions=None):
        return [{"key": k, "x": self.curves[k][0], "si": self.curves[k][1]}
                for k in self.node_keys(conditions)]

    def conductance_levels(self, n_levels, vg_min, vg_max):
        """n_levels gate voltages evenly spaced in the window -> channel conductance G = Id/Vds."""
        lo, hi = max(vg_min, self.vg.min()), min(vg_max, self.vg.max())
        vg_lv = np.linspace(lo, hi, int(n_levels))
        log_id = PchipInterpolator(self.vg, np.log10(np.abs(self.id_a)))(vg_lv)
        g = 10.0 ** log_id / self.vds
        return vg_lv, g


def load_device_library(device_dir, read_vds):
    device_dir = Path(device_dir)
    si_path = device_dir / "sensitization_index.csv"
    tr_path = device_dir / "transfer_curves.csv"
    for p in (si_path, tr_path):
        if not p.exists():
            sys.exit(f"[error] device file not found: {p}")
    si = pd.read_csv(si_path)
    tr = pd.read_csv(tr_path)
    need_si = {"case", "condition", "variable", "x", "SI"}
    need_tr = {"case", "Vds_V", "Vg_V", "Id_A"}
    if not need_si <= set(si.columns):
        sys.exit(f"[error] {si_path.name} needs columns {sorted(need_si)}")
    if not need_tr <= set(tr.columns):
        sys.exit(f"[error] {tr_path.name} needs columns {sorted(need_tr)}")
    si = si.dropna(subset=["x", "SI"])
    lib = {}
    for case, g in si.groupby("case", sort=False):
        curves = {}
        for (cond, var), c in g.groupby(["condition", "variable"], sort=False):
            c = c.groupby("x", as_index=False)["SI"].mean().sort_values("x")
            if len(c) < 3:
                print(f"[warn] {case}: curve {cond}/{var} has <3 points - skipped")
                continue
            curves[(cond, var)] = (c["x"].to_numpy(float), c["SI"].to_numpy(float))
        t = tr[(tr["case"] == case) & np.isclose(tr["Vds_V"], read_vds)].sort_values("Vg_V")
        if t.empty:
            print(f"[warn] {case}: no transfer curve at Vds = {read_vds} V - case skipped")
            continue
        t = t.groupby("Vg_V", as_index=False)["Id_A"].mean()
        lib[case] = DeviceCase(
            name=case,
            zro2_um=float(g["zro2_um"].iloc[0]) if "zro2_um" in g else np.nan,
            mgo_um=float(g["mgo_um"].iloc[0]) if "mgo_um" in g else np.nan,
            curves=curves, vg=t["Vg_V"].to_numpy(float), id_a=t["Id_A"].to_numpy(float),
            vds=float(read_vds))
    if not lib:
        sys.exit("[error] no usable device case found in the device CSV files")
    return lib


# ============================================================================
# 2. Model building blocks
# ============================================================================
class SensitizationReservoir(BaseEstimator, TransformerMixin):
    """Physical (untrained) layer built from the measured sensitization curves.

    Every input feature x in [0, 1] is applied to the device as a paired-pulse
    stimulus whose swept parameter (interval / pulse width / gate amplitude) is
    set linearly by x inside the measured window of that curve.  The node state
    is the measured sensitization index SI = A2/A1 at that parameter
    (PCHIP interpolation between measured points; no over-shoot).
    d features  ->  d * K node states   (K = number of curves, 8 here).
    """

    def __init__(self, curves=None):
        self.curves = curves

    def fit(self, X, y=None):
        self.funcs_ = []
        for c in self.curves:
            x = np.asarray(c["x"], float)
            s = np.asarray(c["si"], float)
            self.funcs_.append((PchipInterpolator(x, s), float(x.min()), float(x.max())))
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.clip(np.asarray(X, float), 0.0, 1.0)
        n, d = X.shape
        out = np.empty((n, d * len(self.funcs_)))
        k = 0
        for j in range(d):
            for f, lo, hi in self.funcs_:
                p = np.clip(lo + X[:, j] * (hi - lo), lo, hi)
                out[:, k] = f(p)
                k += 1
        return out

    def node_names(self, feature_names):
        keys = [c["key"] for c in self.curves]
        return [f"{f}|{k[0]}|{k[1]}" for f in feature_names for k in keys]


def fit_logreg_cv(Z, y, Cs, class_weight, seed, max_iter=5000):
    """L2 logistic regression; C picked by inner stratified CV on log-loss."""
    Cs = list(Cs)
    counts = np.bincount(y)
    n_inner = int(min(3, counts[counts > 0].min()))
    if len(Cs) > 1 and n_inner >= 2:
        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
        best_c, best = Cs[0], -np.inf
        for C in Cs:
            scores = []
            for tr, va in inner.split(Z, y):
                m = LogisticRegression(C=C, max_iter=max_iter, class_weight=class_weight)
                m.fit(Z[tr], y[tr])
                scores.append(-log_loss(y[va], m.predict_proba(Z[va]), labels=m.classes_))
            if np.mean(scores) > best:
                best_c, best = C, np.mean(scores)
    else:
        best_c = Cs[len(Cs) // 2]
    return LogisticRegression(C=best_c, max_iter=max_iter, class_weight=class_weight).fit(Z, y)


def program_weights(W, g_levels, g_min, g_max, noise=0.0, rng=None):
    """Store float weights W (n_out x n_in) in differential device pairs.

    Each weight uses two devices: w = s * (G+ - G-) / (Gmax - Gmin).
    The magnitude |w|/s is rounded to the nearest available conductance level
    (levels come from the transfer curve, so they are NOT evenly spaced) and the
    partner device sits at Gmin.  s = max|w| of that output neuron (its gain).
    `noise` = relative Gaussian spread of every programmed conductance.
    """
    W = np.atleast_2d(np.asarray(W, float))
    g_levels = np.asarray(g_levels, float)
    span = g_max - g_min
    gn = (g_levels - g_min) / span
    rng = rng if rng is not None else np.random.default_rng(0)
    out = np.zeros_like(W)
    for r in range(W.shape[0]):
        s = np.max(np.abs(W[r]))
        if s == 0:
            continue
        t = np.abs(W[r]) / s
        idx = np.abs(t[:, None] - gn[None, :]).argmin(axis=1)
        g_sel = g_levels[idx]
        g_pos = np.where(W[r] >= 0, g_sel, g_min)
        g_neg = np.where(W[r] >= 0, g_min, g_sel)
        if noise > 0:
            g_pos = np.clip(g_pos * (1 + noise * rng.standard_normal(g_pos.shape)), 0, None)
            g_neg = np.clip(g_neg * (1 + noise * rng.standard_normal(g_neg.shape)), 0, None)
        out[r] = s * (g_pos - g_neg) / span
    return out


def _proba_from_decision(dec):
    if dec.shape[1] == 1:                           # binary
        p = 1.0 / (1.0 + np.exp(-dec[:, 0]))
        return np.column_stack([1 - p, p])
    dec = dec - dec.max(axis=1, keepdims=True)      # multinomial softmax
    e = np.exp(dec)
    return e / e.sum(axis=1, keepdims=True)


class LogRegCV(BaseEstimator, ClassifierMixin):
    """Software reference: standardise + L2 logistic regression (same C grid)."""

    def __init__(self, Cs=(0.01, 0.1, 1, 10), class_weight="balanced", seed=0):
        self.Cs = Cs
        self.class_weight = class_weight
        self.seed = seed

    def fit(self, X, y):
        self.scaler_ = StandardScaler().fit(X)
        self.lr_ = fit_logreg_cv(self.scaler_.transform(X), y, self.Cs, self.class_weight, self.seed)
        self.classes_ = self.lr_.classes_
        return self

    def predict_proba(self, X):
        return self.lr_.predict_proba(self.scaler_.transform(X))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class MLPCV(BaseEstimator, ClassifierMixin):
    """Software reference ANN: standardise + 1-hidden-layer MLP (L-BFGS);
    L2 strength alpha picked by inner stratified CV (works for small cohorts)."""

    def __init__(self, hidden=64, alphas=(0.1, 1.0, 10.0), seed=0, max_iter=300):
        self.hidden = hidden
        self.alphas = alphas
        self.seed = seed
        self.max_iter = max_iter

    def _mlp(self, alpha):
        return MLPClassifier(hidden_layer_sizes=(self.hidden,), solver="lbfgs", alpha=alpha,
                             max_iter=self.max_iter, random_state=self.seed)

    def fit(self, X, y):
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        counts = np.bincount(y)
        n_inner = int(min(3, counts[counts > 0].min()))
        best_a, best = self.alphas[len(self.alphas) // 2], -np.inf
        if len(self.alphas) > 1 and n_inner >= 2:
            inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=self.seed)
            for a in self.alphas:
                sc = []
                for tr, va in inner.split(Xs, y):
                    m = self._mlp(a).fit(Xs[tr], y[tr])
                    sc.append(-log_loss(y[va], m.predict_proba(Xs[va]), labels=m.classes_))
                if np.mean(sc) > best:
                    best_a, best = a, np.mean(sc)
        self.alpha_ = best_a
        self.mlp_ = self._mlp(best_a).fit(Xs, y)
        self.classes_ = self.mlp_.classes_
        return self

    def predict_proba(self, X):
        return self.mlp_.predict_proba(self.scaler_.transform(X))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class DeviceSynapseReadout(BaseEstimator, ClassifierMixin):
    """Linear readout whose weights are stored in device conductances.

    fit(): per-node standardisation (peripheral gain/offset) -> logistic regression
    trained in software -> weights programmed into the device levels.
    predict_proba(weights="device") uses the programmed weights (default);
    weights="float" gives the ideal software readout for comparison.
    The bias is kept in the peripheral circuit (not quantised).
    """

    def __init__(self, g_levels=None, g_min=None, g_max=None, noise=0.0,
                 Cs=(0.01, 0.1, 1, 10), class_weight="balanced", seed=0):
        self.g_levels = g_levels
        self.g_min = g_min
        self.g_max = g_max
        self.noise = noise
        self.Cs = Cs
        self.class_weight = class_weight
        self.seed = seed

    def fit(self, Z, y):
        self.scaler_ = StandardScaler().fit(Z)
        lr = fit_logreg_cv(self.scaler_.transform(Z), y, self.Cs, self.class_weight, self.seed)
        self.C_ = lr.C
        self.classes_ = lr.classes_
        self.coef_float_ = lr.coef_.copy()
        self.intercept_ = lr.intercept_.copy()
        self.coef_device_ = program_weights(self.coef_float_, self.g_levels, self.g_min, self.g_max,
                                            noise=self.noise, rng=np.random.default_rng(self.seed))
        return self

    def proba_with(self, Z, W):
        return _proba_from_decision(self.scaler_.transform(Z) @ np.atleast_2d(W).T + self.intercept_)

    def predict_proba(self, Z, weights="device"):
        W = self.coef_device_ if weights == "device" else self.coef_float_
        return self.proba_with(Z, W)

    def predict(self, Z, weights="device"):
        return self.classes_[self.predict_proba(Z, weights).argmax(1)]


# ============================================================================
# 3. Patient CSV handling
# ============================================================================
ID_PATTERN = re.compile(r"(^id$|^id[\s_]|[\s_]id$|patient.?id|subject.?id|mri.?id|record.?id|"
                        r"^name$|doctor)", re.I)


PASS_ID = re.compile(r"(^id$|^id[\s_]|[\s_]id$|patient.?id|subject.?id|mri.?id|record.?id)", re.I)


def looks_like_id(name, s):
    if ID_PATTERN.search(str(name)):
        return True
    n = s.notna().sum()
    if not pd.api.types.is_numeric_dtype(s) and n > 20 and s.nunique() > 0.95 * n:
        return True
    return False


def infer_columns(df, target, drop, group_col, max_missing, max_categories):
    num, cat, dropped = [], [], []
    for c in df.columns:
        if c == target or c == group_col or c in drop:
            continue
        s = df[c]
        if s.isna().mean() > max_missing:
            dropped.append((c, f">{int(max_missing * 100)}% missing"))
            continue
        if s.nunique(dropna=True) <= 1:
            dropped.append((c, "constant"))
            continue
        if looks_like_id(c, s):
            dropped.append((c, "identifier / free text"))
            continue
        if pd.api.types.is_bool_dtype(s):
            cat.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            num.append(c)
        else:
            conv = pd.to_numeric(s, errors="coerce")
            if conv.notna().sum() >= 0.9 * s.notna().sum():
                df[c] = conv
                num.append(c)
            elif s.nunique(dropna=True) <= max_categories:
                df[c] = s.astype("object").where(s.notna(), np.nan)
                cat.append(c)
            else:
                dropped.append((c, f"text with {s.nunique()} categories"))
    return num, cat, dropped


def _onehot():
    for kw in ({"sparse_output": False, "drop": "if_binary"}, {"sparse": False, "drop": "if_binary"},
               {"sparse": False}):
        try:
            enc = OneHotEncoder(handle_unknown="ignore", **kw)
            enc.fit(np.array([["a"], ["b"]], dtype=object))  # old versions reject drop + ignore
            return OneHotEncoder(handle_unknown="ignore", **kw)
        except (TypeError, ValueError):
            continue
    return OneHotEncoder(handle_unknown="ignore")


def make_preprocessor(num_cols, cat_cols):
    parts = []
    if num_cols:
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("scale", MinMaxScaler(clip=True))]), num_cols))
    if cat_cols:
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", _onehot())]), cat_cols))
    return ColumnTransformer(parts, remainder="drop")


def prepare_target(df, target, positive):
    if target not in df.columns:
        sys.exit(f"[error] target column '{target}' not in CSV. Columns: {list(df.columns)}")
    raw = df[target]
    mask = raw.notna().to_numpy()
    raw = raw[mask]
    if positive:
        pos = [p.strip() for p in str(positive).split(",") if p.strip()]

        def is_pos(v):
            sv = str(v).strip()
            if sv in pos:
                return True
            try:
                return any(np.isclose(float(sv), float(p)) for p in pos)
            except ValueError:
                return False
        y = np.array([1 if is_pos(v) else 0 for v in raw])
        names = [f"not({','.join(pos)})", ",".join(pos)]
        if y.min() == y.max():
            sys.exit(f"[error] --positive {positive} gives a single class; check the labels "
                     f"{sorted(raw.astype(str).unique())[:10]}")
    else:
        le = LabelEncoder().fit(raw.astype(str))
        y = le.transform(raw.astype(str))
        names = list(le.classes_)
    return mask, y, names


def leakage_check(X, y, num_cols):
    """Warn about single columns that almost perfectly predict the label (e.g. CDR)."""
    if len(np.unique(y)) != 2:
        return []
    hits = []
    for c in num_cols:
        v = pd.to_numeric(X[c], errors="coerce")
        ok = v.notna().to_numpy()
        if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
            continue
        auc = roc_auc_score(y[ok], v[ok])
        auc = max(auc, 1 - auc)
        if auc > 0.97:
            hits.append((c, auc))
    return hits


# ============================================================================
# 4. Model zoo
# ============================================================================
def build_models(lib, pre, args, ablation=False):
    Cs = tuple(float(c) for c in args.Cs.split(","))
    models, meta = {}, {}
    models["software_logreg"] = Pipeline([("pre", pre), ("clf", LogRegCV(Cs, args.class_weight, args.seed))])
    meta["software_logreg"] = {"kind": "software", "case": None}
    if not args.skip_mlp:
        models["software_mlp"] = Pipeline([("pre", pre), ("clf", MLPCV(64, seed=args.seed))])
        meta["software_mlp"] = {"kind": "software", "case": None}
    for case, dev in lib.items():
        _, g = dev.conductance_levels(args.n_levels, args.vg_min, args.vg_max)
        variants = [(case, None)]
        if ablation:
            variants += [(f"{case}__only_{c}", [c]) for c in CONDITIONS
                         if any(k[0] == c for k in dev.curves)]
        for name, conds in variants:
            models[name] = Pipeline([
                ("pre", pre),
                ("reservoir", SensitizationReservoir(dev.reservoir_curves(conds))),
                ("readout", DeviceSynapseReadout(g_levels=g, g_min=g.min(), g_max=g.max(),
                                                 noise=args.noise, Cs=Cs,
                                                 class_weight=args.class_weight, seed=args.seed)),
            ])
            meta[name] = {"kind": "device", "case": case,
                          "conditions": conds or sorted({k[0] for k in dev.curves}),
                          "ablation": conds is not None}
    return models, meta


# ============================================================================
# 5. Metrics
# ============================================================================
def compute_metrics(y, proba):
    pred = proba.argmax(1)
    m = {"accuracy": accuracy_score(y, pred),
         "balanced_accuracy": balanced_accuracy_score(y, pred),
         "f1_macro": f1_score(y, pred, average="macro", zero_division=0)}
    if proba.shape[1] == 2:
        m["sensitivity"] = recall_score(y, pred, pos_label=1, zero_division=0)
        m["specificity"] = recall_score(y, pred, pos_label=0, zero_division=0)
        m["precision"] = precision_score(y, pred, pos_label=1, zero_division=0)
        try:
            m["roc_auc"] = roc_auc_score(y, proba[:, 1])
        except ValueError:
            m["roc_auc"] = np.nan
    else:
        try:
            m["roc_auc"] = roc_auc_score(y, proba, multi_class="ovr", average="macro",
                                         labels=np.arange(proba.shape[1]))
        except ValueError:
            m["roc_auc"] = np.nan
    return m


def _full_proba(model_classes, proba, n_classes):
    """Expand probabilities if a training fold missed a class."""
    if proba.shape[1] == n_classes:
        return proba
    full = np.zeros((proba.shape[0], n_classes))
    full[:, np.asarray(model_classes)] = proba
    return full


# ============================================================================
# 6. Figures (matplotlib, light surface; entity colours stay fixed everywhere)
# ============================================================================
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CASE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SOFT_COLORS = {"software_logreg": "#898781", "software_mlp": "#52514e"}
PRETTY = {"software_logreg": "Software: logistic regression", "software_mlp": "Software: MLP (64 hidden)"}


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.labelcolor": INK2, "font.size": 10, "lines.linewidth": 2,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def make_figures(out, lib, case_colors, summary, oof, y, class_names, cms, noise_df, levels_df,
                 abl_df, args, base_models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    main = base_models
    color = {**SOFT_COLORS, **case_colors}
    label = {**PRETTY, **{c: f"Device: {lib[c].label}" for c in lib}}

    # (a) model comparison: dot + sd (a dot plot may start above zero) --------
    s = summary[(summary.readout.isin(["software", "device"])) & summary.model.isin(main)]
    s = s.set_index("model").loc[[m for m in main if m in set(s.model)]].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(s) + 1.8), sharey=True)
    for ax, met, title in zip(axes, ["balanced_accuracy", "roc_auc"], ["Balanced accuracy", "ROC-AUC"]):
        ypos = np.arange(len(s))[::-1]
        mean, sd = s[f"{met}_mean"].to_numpy(float), s[f"{met}_std"].to_numpy(float)
        for yp, m, v, e in zip(ypos, s.model, mean, sd):
            if not np.isfinite(v):
                continue
            ax.errorbar(v, yp, xerr=e, fmt="o", ms=9, color=color[m], mec=SURFACE, mew=2,
                        ecolor=color[m], elinewidth=2, capsize=0)
            ax.text(v, yp + 0.22, f"{v:.3f}", ha="center", va="bottom", color=INK, fontsize=9)
        lo = np.nanmin(mean - np.nan_to_num(sd))
        ax.set_xlim(max(0.0, np.floor((lo - 0.03) * 20) / 20), 1.0)
        ax.set_ylim(-0.6, len(s) - 0.3)
        rep_txt = f" x {args.repeats}" if args.repeats > 1 else ""
        ax.set_title(f"{title} ({args.folds}-fold CV{rep_txt}, mean ± sd)", loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_yticks(ypos)
        ax.set_yticklabels([label[m] for m in s.model])
    fig.suptitle("Alzheimer's prediction: device models vs software references "
                 f"(device weights: {args.n_levels} levels, noise {args.noise:g})",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_model_comparison.png", dpi=200)
    plt.close(fig)

    # (b) ROC curves (binary only) ---------------------------------------
    if len(class_names) == 2:
        fig, ax = plt.subplots(figsize=(6.2, 5.6))
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
        for m in main:
            col = f"{m}__p1"
            if col not in oof:
                continue
            fpr, tpr, _ = roc_curve(y, oof[col])
            auc = roc_auc_score(y, oof[col])
            ax.plot(fpr, tpr, color=color[m], lw=2, label=f"{label[m]}  (AUC {auc:.3f})")
        ax.set_xlabel("False positive rate (1 - specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.set_title(f"ROC, out-of-fold predictions  [positive = {class_names[1]}]", loc="left")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig2_roc_curves.png", dpi=200)
        plt.close(fig)

    # (c) device node transfer functions (small multiples) ----------------
    keys = []
    for dev in lib.values():
        for k in dev.node_keys():
            if k not in keys:
                keys.append(k)
    ncol = 4
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.7 * nrow), squeeze=False)
    u = np.linspace(0, 1, 200)
    for ax, k in zip(axes.flat, keys):
        for case, dev in lib.items():
            if k not in dev.curves:
                continue
            x, si = dev.curves[k]
            f = PchipInterpolator(x, si)
            ax.plot(u, f(x.min() + u * (x.max() - x.min())), color=case_colors[case], lw=2)
            ax.plot((x - x.min()) / (x.max() - x.min()), si, "o", ms=5, color=case_colors[case],
                    mec=SURFACE, mew=1.5)
        ax.set_title(f"{CONDITION_LABEL.get(k[0], k[0])} · {VARIABLE_LABEL.get(k[1], k[1])}",
                     loc="left", fontsize=10)
        ax.set_xlabel("normalised input x", fontsize=9)
        ax.set_ylabel("SI = A2/A1", fontsize=9)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=case_colors[c], lw=2, marker="o", mec=SURFACE, label=lib[c].label)
               for c in lib]
    fig.legend(handles=handles, loc="lower center", ncol=len(lib), fontsize=9)
    fig.suptitle("Reservoir nodes = measured sensitization responses (dots) + PCHIP interpolation (lines)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(fig_dir / "fig3_device_nodes.png", dpi=200)
    plt.close(fig)

    # (d) conductance levels ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for case, dev in lib.items():
        vg_l, g = dev.conductance_levels(args.n_levels, args.vg_min, args.vg_max)
        m = (dev.vg >= args.vg_min - 1) & (dev.vg <= args.vg_max + 1)
        axes[0].plot(dev.vg[m], np.abs(dev.id_a[m]) / dev.vds, color=case_colors[case], lw=2,
                     label=dev.label)
        axes[0].plot(vg_l, g, "o", ms=4, color=case_colors[case], mec=SURFACE, mew=1)
        gn = (g - g.min()) / (g.max() - g.min())
        axes[1].plot(np.arange(len(gn)), gn, "o-", ms=5, lw=1.5, color=case_colors[case],
                     mec=SURFACE, mew=1.5, label=dev.label)
    axes[0].set_yscale("log")
    axes[0].axvspan(args.vg_min, args.vg_max, color=GRID, alpha=0.35, lw=0)
    axes[0].set_xlabel("Gate voltage Vg (V)")
    axes[0].set_ylabel(f"Conductance G = Id/Vds (S), Vds = {args.read_vds:g} V")
    axes[0].set_title("Synaptic weight states from the transfer curve", loc="left")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Level index (evenly spaced Vg)")
    axes[1].set_ylabel("Normalised weight (G - Gmin)/(Gmax - Gmin)")
    axes[1].set_title(f"{args.n_levels} programmable levels per device", loc="left")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_conductance_levels.png", dpi=200)
    plt.close(fig)

    # (e) robustness -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    base = summary[(summary.model == "software_logreg")]["balanced_accuracy_mean"]
    for ax in axes:
        if len(base):
            ax.axhline(float(base.iloc[0]), color=MUTED, lw=1.5)
    nd = noise_df.groupby(["case", "noise"], as_index=False)["balanced_accuracy"].mean()
    ld = levels_df.groupby(["case", "n_levels"], as_index=False)["balanced_accuracy"].mean()
    for case in lib:
        d = nd[nd.case == case]
        axes[0].plot(100 * d.noise, d.balanced_accuracy, "o-", color=case_colors[case], ms=6,
                     mec=SURFACE, mew=1.5, label=lib[case].label)
        d = ld[ld.case == case]
        axes[1].plot(d.n_levels, d.balanced_accuracy, "o-", color=case_colors[case], ms=6,
                     mec=SURFACE, mew=1.5, label=lib[case].label)
    axes[0].set_xlabel("Conductance variation σ (%)")
    axes[0].set_ylabel("Balanced accuracy (CV mean)")
    axes[0].set_title(f"Robustness to device variation ({args.n_levels} levels)", loc="left")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(sorted(ld.n_levels.unique()))
    axes[1].set_xticklabels([str(int(v)) for v in sorted(ld.n_levels.unique())])
    axes[1].set_xlabel("Conductance levels per device")
    axes[1].set_title("Effect of weight resolution (no noise)", loc="left")
    axes[0].legend(fontsize=8, loc="lower left")
    if len(base):
        axes[1].text(axes[1].get_xlim()[0], float(base.iloc[0]), "  software logistic regression",
                     va="bottom", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_robustness.png", dpi=200)
    plt.close(fig)

    # (f) confusion matrices ---------------------------------------------
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#f4f8fe", "#86b6ef", "#2a78d6", "#104281"])
    ms = [m for m in main if m in cms]
    fig, axes = plt.subplots(1, len(ms), figsize=(3.1 * len(ms), 3.9), squeeze=False)
    for ax, m in zip(axes[0], ms):
        cm = cms[m]
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=10,
                    color="white" if v > 0.55 * cm.max() else INK)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        short = [c if len(c) <= 12 else c[:11] + "…" for c in class_names]
        ax.set_xticklabels(short, fontsize=8)
        ax.set_yticklabels(short, fontsize=8)
        ax.grid(False)
        ax.set_xlabel("predicted", fontsize=9)
        ax.set_ylabel("true", fontsize=9)
        ttl = label[m].replace("Device: ", "").replace("Software: ", "SW: ")
        ax.set_title(ttl, fontsize=9, loc="left")
    fig.suptitle("Confusion matrices (summed over CV folds)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(fig_dir / "fig6_confusion_matrices.png", dpi=200)
    plt.close(fig)

    # (g) ablation ---------------------------------------------------------
    if abl_df is not None and len(abl_df):
        cases = list(lib)
        fig, axes = plt.subplots(1, len(cases), figsize=(3.8 * len(cases), 3.8), sharey=True, squeeze=False)
        order = ["all"] + CONDITIONS
        for ax, case in zip(axes[0], cases):
            d = abl_df[abl_df.case == case].set_index("nodes").reindex(order).dropna(how="all")
            xs = np.arange(len(d))
            ax.bar(xs, d.balanced_accuracy_mean, width=0.5, color=case_colors[case],
                   yerr=d.balanced_accuracy_std, error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 3})
            for xx, v, e in zip(xs, d.balanced_accuracy_mean, d.balanced_accuracy_std.fillna(0)):
                ax.text(xx, v + e + 0.015, f"{v:.3f}", ha="center", va="bottom", color=INK, fontsize=8)
            ax.set_xticks(xs)
            ax.set_xticklabels(["all 8 nodes" if i == "all" else "only " + CONDITION_LABEL[i]
                                for i in d.index], rotation=20, fontsize=8)
            ax.set_title(lib[case].label, loc="left", fontsize=10)
            ax.grid(axis="x", visible=False)
            ax.set_ylim(0, 1.08)
        axes[0][0].set_ylabel("Balanced accuracy (device weights)")
        fig.suptitle("Which stimulus condition carries the information?", x=0.01, ha="left",
                     color=INK, fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig7_condition_ablation.png", dpi=200)
        plt.close(fig)


# ============================================================================
# 7. Training / evaluation driver
# ============================================================================
def summarise(per_fold):
    metric_cols = [c for c in per_fold.columns if c not in ("model", "readout", "fold", "case")]
    g = per_fold.groupby(["model", "readout"], sort=False)[metric_cols]
    mean, std = g.mean().add_suffix("_mean"), g.std(ddof=0).add_suffix("_std")
    out = pd.concat([mean, std], axis=1).reset_index()
    cols = ["model", "readout"] + [x for m in metric_cols for x in (f"{m}_mean", f"{m}_std")]
    return out[cols]


def train(args):
    t0 = time.time()
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    lib = load_device_library(args.device_dir, args.read_vds)
    case_colors = {c: CASE_COLORS[i % len(CASE_COLORS)] for i, c in enumerate(lib)}

    df = pd.read_csv(args.data, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    drop = [c.strip() for c in args.drop.split(",")] if args.drop else []
    mask, y, class_names = prepare_target(df, args.target, args.positive)
    df = df.loc[mask].reset_index(drop=True)
    groups = df[args.group_col].to_numpy() if args.group_col else None
    num_cols, cat_cols, dropped = infer_columns(df, args.target, drop, args.group_col,
                                                args.max_missing, args.max_categories)
    feat_cols = num_cols + cat_cols
    if not feat_cols:
        sys.exit("[error] no usable feature columns left")
    X = df[feat_cols].copy()

    print("=" * 78)
    print(f"Patients: {len(df)}   target: '{args.target}'   classes: "
          + ", ".join(f"{n} = {int((y == i).sum())}" for i, n in enumerate(class_names)))
    print(f"Features used ({len(feat_cols)}): numeric {num_cols}  categorical {cat_cols}")
    if drop:
        print(f"Dropped by --drop: {drop}")
    for c, why in dropped:
        print(f"Dropped automatically: {c}  ({why})")
    leaks = leakage_check(X, y, num_cols)
    for c, auc in leaks:
        print(f"[WARNING] '{c}' alone predicts the label with AUC {auc:.3f} - if it is part of "
              f"the diagnosis itself (e.g. CDR), remove it with --drop {c}")
    print(f"Device cases (models): {', '.join(f'{c} [{lib[c].label_ascii}]' for c in lib)}")
    n_enc = clone(make_preprocessor(num_cols, cat_cols)).fit(X).transform(X.head(2)).shape[1]
    k_nodes = len(next(iter(lib.values())).curves)
    print(f"Reservoir: {k_nodes} nodes per input -> {n_enc} encoded inputs give {n_enc * k_nodes} node states")
    print(f"Synapses: {args.n_levels} levels, Vg {args.vg_min}..{args.vg_max} V @ Vds {args.read_vds} V, "
          f"noise sigma {args.noise:g}")
    print("=" * 78)

    pre = make_preprocessor(num_cols, cat_cols)
    models, meta = build_models(lib, pre, args, ablation=args.ablation)
    base_models = [m for m in models if not meta[m].get("ablation")]

    n_classes = len(class_names)
    min_class = np.bincount(y).min()
    n_splits = int(min(args.folds, min_class))
    if n_splits < 2:
        sys.exit("[error] a class has fewer than 2 patients - cannot cross-validate")
    splits = []
    for rep in range(max(1, args.repeats)):
        if groups is not None:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed + rep)
            splits += [(rep, tr, te) for tr, te in splitter.split(X, y, groups)]
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed + rep)
            splits += [(rep, tr, te) for tr, te in splitter.split(X, y)]
    args.folds = n_splits

    noise_grid = [float(v) for v in args.noise_grid.split(",")]
    levels_grid = [int(v) for v in args.levels_grid.split(",")]
    rows, noise_rows, level_rows = [], [], []
    oof = {}
    cms = {}
    for fold, (rep, tr, te) in enumerate(splits):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
        for name, est in models.items():
            m = clone(est).fit(Xtr, ytr)
            if meta[name]["kind"] == "software":
                P = {"software": _full_proba(m.classes_, m.predict_proba(Xte), n_classes)}
            else:
                Z = m[:-1].transform(Xte)
                ro = m[-1]
                P = {"float": _full_proba(ro.classes_, ro.predict_proba(Z, "float"), n_classes),
                     "device": _full_proba(ro.classes_, ro.predict_proba(Z, "device"), n_classes)}
            for readout, proba in P.items():
                rows.append({"model": name, "readout": readout, "fold": fold,
                             **compute_metrics(yte, proba)})
            key = "software" if meta[name]["kind"] == "software" else "device"
            if name in base_models and rep == 0:     # OOF / confusion from the first repeat
                pr = P[key]
                for k in range(n_classes):
                    col = f"{name}__p{k}"
                    oof.setdefault(col, np.full(len(y), np.nan))[te] = pr[:, k]
                cms[name] = cms.get(name, 0) + confusion_matrix(yte, pr.argmax(1), labels=range(n_classes))
            # --- robustness sweeps (main device models only) ---
            if meta[name]["kind"] == "device" and not meta[name]["ablation"]:
                dev = lib[meta[name]["case"]]
                _, g = dev.conductance_levels(args.n_levels, args.vg_min, args.vg_max)
                rng = np.random.default_rng(args.seed + 1000 * fold)
                for sigma in noise_grid:
                    reps = 1 if sigma == 0 else args.noise_repeats
                    for r in range(reps):
                        W = program_weights(ro.coef_float_, g, g.min(), g.max(), sigma, rng)
                        pr = _full_proba(ro.classes_, ro.proba_with(Z, W), n_classes)
                        mm = compute_metrics(yte, pr)
                        noise_rows.append({"case": dev.name, "fold": fold, "noise": sigma, "repeat": r,
                                           "accuracy": mm["accuracy"],
                                           "balanced_accuracy": mm["balanced_accuracy"],
                                           "roc_auc": mm["roc_auc"]})
                for nl in levels_grid:
                    _, g2 = dev.conductance_levels(nl, args.vg_min, args.vg_max)
                    W = program_weights(ro.coef_float_, g2, g2.min(), g2.max(), 0.0)
                    pr = _full_proba(ro.classes_, ro.proba_with(Z, W), n_classes)
                    mm = compute_metrics(yte, pr)
                    level_rows.append({"case": dev.name, "fold": fold, "n_levels": nl,
                                       "accuracy": mm["accuracy"],
                                       "balanced_accuracy": mm["balanced_accuracy"],
                                       "roc_auc": mm["roc_auc"]})
        print(f"  fold {fold + 1}/{len(splits)} done  ({time.time() - t0:.0f} s)")

    per_fold = pd.DataFrame(rows)
    summary = summarise(per_fold)
    noise_df, levels_df = pd.DataFrame(noise_rows), pd.DataFrame(level_rows)
    per_fold.to_csv(out / "metrics_per_fold.csv", index=False)
    summary.to_csv(out / "metrics_summary.csv", index=False, float_format="%.4f")
    for frame, key, fname in [(noise_df, "noise", "robustness_noise.csv"),
                              (levels_df, "n_levels", "robustness_levels.csv")]:
        agg = frame.groupby(["case", key])[["accuracy", "balanced_accuracy", "roc_auc"]].agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg.reset_index().to_csv(out / fname, index=False, float_format="%.4f")

    abl_df = None
    if args.ablation:
        recs = []
        for name, mt in meta.items():
            if mt["kind"] != "device":
                continue
            r = summary[(summary.model == name) & (summary.readout == "device")].iloc[0]
            recs.append({"case": mt["case"],
                         "nodes": "all" if not mt["ablation"] else mt["conditions"][0],
                         "n_curves": len(models[name].named_steps["reservoir"].curves),
                         "balanced_accuracy_mean": r.balanced_accuracy_mean,
                         "balanced_accuracy_std": r.balanced_accuracy_std,
                         "roc_auc_mean": r.roc_auc_mean, "accuracy_mean": r.accuracy_mean})
        abl_df = pd.DataFrame(recs)
        abl_df.to_csv(out / "ablation_conditions.csv", index=False, float_format="%.4f")

    oof_df = pd.DataFrame({"row": np.arange(len(y)), "true_label": [class_names[i] for i in y]})
    for k, v in oof.items():
        oof_df[k] = v
    oof_df.to_csv(out / "oof_predictions.csv", index=False, float_format="%.5f")

    lv = []
    for case, dev in lib.items():
        vg_l, g = dev.conductance_levels(args.n_levels, args.vg_min, args.vg_max)
        gn = (g - g.min()) / (g.max() - g.min())
        for i, (a, b, c) in enumerate(zip(vg_l, g, gn)):
            lv.append({"case": case, "level": i, "Vg_V": a, "G_S": b, "G_normalised": c})
    pd.DataFrame(lv).to_csv(out / "device_levels.csv", index=False, float_format="%.6g")

    # ---- final models on all patients (for prediction) ----
    for name in base_models:
        final = clone(models[name]).fit(X, y)
        bundle = {"model": final, "kind": meta[name]["kind"], "case": meta[name]["case"],
                  "class_names": class_names, "target": args.target, "positive": args.positive,
                  "feature_columns": feat_cols, "numeric_columns": num_cols,
                  "categorical_columns": cat_cols, "settings": vars(args).copy(),
                  "created": time.strftime("%Y-%m-%d %H:%M"), "sklearn_version": __import__("sklearn").__version__}
        joblib.dump(bundle, out / "models" / f"{name}.joblib")

    # ---- console table + report ----
    show = summary[summary.model.isin(base_models)].copy()
    show["model"] = show.apply(lambda r: r.model if r.readout == "software" else f"{r.model} [{r.readout} w]", axis=1)
    cols = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1_macro", "roc_auc"]
    cols = [c for c in cols if f"{c}_mean" in show]
    table = pd.DataFrame({"model": show.model})
    for c in cols:
        table[c] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(show[f"{c}_mean"], show[f"{c}_std"])]
    print("\n" + table.to_string(index=False))
    nsum = noise_df.groupby(["case", "noise"])["balanced_accuracy"].mean().unstack()
    print("\nBalanced accuracy vs conductance noise sigma:\n" + nsum.round(3).to_string())
    lsum = levels_df.groupby(["case", "n_levels"])["balanced_accuracy"].mean().unstack()
    print("\nBalanced accuracy vs number of levels:\n" + lsum.round(3).to_string())
    if abl_df is not None:
        print("\nCondition ablation (device weights, balanced accuracy):\n"
              + abl_df.pivot(index="case", columns="nodes", values="balanced_accuracy_mean").round(3).to_string())

    with open(out / "report.md", "w", encoding="utf-8") as f:
        f.write(f"# Device-aware Alzheimer's prediction - results\n\n")
        f.write(f"- data: `{Path(args.data).name}`, {len(df)} patients, target `{args.target}` "
                f"({', '.join(f'{n}: {int((y == i).sum())}' for i, n in enumerate(class_names))})\n")
        f.write(f"- features ({len(feat_cols)}): {', '.join(feat_cols)}\n")
        f.write(f"- {n_splits}-fold {'grouped ' if groups is not None else ''}stratified CV"
                f"{f' x {args.repeats} repeats' if args.repeats > 1 else ''}, seed {args.seed}\n")
        f.write(f"- synapses: {args.n_levels} levels in Vg {args.vg_min}..{args.vg_max} V at Vds "
                f"{args.read_vds} V, noise sigma {args.noise:g}\n\n")
        f.write("| model | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n")
        for _, r in table.iterrows():
            f.write("| " + " | ".join(str(r[c]) for c in ["model"] + cols) + " |\n")
        f.write("\n`[float w]` = ideal software readout on the device reservoir, "
                "`[device w]` = readout weights stored in device conductance levels.\n")
        f.write("\n## Balanced accuracy vs conductance noise sigma\n\n" + _md_table(nsum.round(3)))
        f.write("\n## Balanced accuracy vs number of conductance levels\n\n" + _md_table(lsum.round(3)))
        if abl_df is not None:
            f.write("\n## Condition ablation (balanced accuracy, device weights)\n\n"
                    + _md_table(abl_df.pivot(index="case", columns="nodes",
                                             values="balanced_accuracy_mean").round(3)))
    cfg = vars(args).copy()
    cfg.update({"python": platform.python_version(), "sklearn": __import__("sklearn").__version__,
                "classes": class_names, "features": feat_cols, "dropped": dropped,
                "device_cases": {c: lib[c].label_ascii for c in lib}})
    with open(out / "run_config.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    if not args.no_plots:
        make_figures(out, lib, case_colors, summary, oof_df, y, class_names, cms, noise_df, levels_df,
                     abl_df, args, base_models)
    print(f"\nAll outputs in: {out.resolve()}   ({time.time() - t0:.0f} s)")


def _md_table(frame):
    frame = frame.reset_index()
    head = "| " + " | ".join(str(c) for c in frame.columns) + " |\n"
    head += "|" + "---|" * len(frame.columns) + "\n"
    body = "".join("| " + " | ".join(str(v) for v in row) + " |\n" for row in frame.itertuples(index=False))
    return head + body


# ============================================================================
# 8. Prediction for new patients
# ============================================================================
def predict(args):
    bundle = joblib.load(args.model)
    df = pd.read_csv(args.data, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in bundle["feature_columns"] if c not in df.columns]
    if missing:
        print(f"[warn] columns missing in new data (filled by training median/mode): {missing}")
    X = df.reindex(columns=bundle["feature_columns"]).copy()
    for c in bundle["numeric_columns"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in bundle["categorical_columns"]:
        X[c] = X[c].astype("object").where(X[c].notna(), np.nan)
    model = bundle["model"]
    if bundle["kind"] == "device":
        proba = model[-1].predict_proba(model[:-1].transform(X), "device")
    else:
        proba = model.predict_proba(X)
    names = bundle["class_names"]
    cls = model.classes_
    res = pd.DataFrame(index=df.index)
    id_cols = [c for c in df.columns if PASS_ID.search(c)]
    for c in id_cols[:1]:
        res[c] = df[c]
    res["predicted_class"] = [names[k] for k in cls[proba.argmax(1)]]
    for j, k in enumerate(cls):
        res[f"P({names[k]})"] = proba[:, j]
    res.to_csv(args.out, index=False, float_format="%.4f")
    print(f"model: {Path(args.model).name}  ({bundle['case'] or 'software'})")
    print(res.head(10).to_string(index=False))
    print(f"... {len(res)} predictions written to {args.out}")


# ============================================================================
# 9. Command line
# ============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="cross-validate all models, save figures, results and final models")
    t.add_argument("--data", required=True, help="patient CSV (one row per patient/visit)")
    t.add_argument("--target", required=True, help="label column, e.g. Diagnosis / Group / CDR")
    t.add_argument("--positive", default=None,
                   help="comma list of label values treated as class 1 (binarises the target), e.g. 0.5,1,2")
    t.add_argument("--drop", default="", help="comma list of columns to ignore (e.g. leakage columns)")
    t.add_argument("--group-col", default=None,
                   help="subject column for longitudinal data (keeps all visits of a subject in one fold)")
    t.add_argument("--device-dir", default=DEFAULTS["device_dir"])
    t.add_argument("--out", default="results")
    t.add_argument("--read-vds", type=float, default=DEFAULTS["read_vds"])
    t.add_argument("--vg-min", type=float, default=DEFAULTS["vg_min"])
    t.add_argument("--vg-max", type=float, default=DEFAULTS["vg_max"])
    t.add_argument("--n-levels", type=int, default=DEFAULTS["n_levels"])
    t.add_argument("--noise", type=float, default=DEFAULTS["noise"],
                   help="relative conductance variation used for the main result (0.05 = 5 %%)")
    t.add_argument("--folds", type=int, default=DEFAULTS["folds"])
    t.add_argument("--repeats", type=int, default=1,
                   help="repeat the K-fold CV with new shuffles (e.g. 5 for small cohorts)")
    t.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    t.add_argument("--Cs", default=DEFAULTS["Cs"], help="L2 grid for the readout")
    t.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])
    t.add_argument("--noise-grid", default=DEFAULTS["noise_grid"])
    t.add_argument("--levels-grid", default=DEFAULTS["levels_grid"])
    t.add_argument("--noise-repeats", type=int, default=DEFAULTS["noise_repeats"])
    t.add_argument("--max-missing", type=float, default=DEFAULTS["max_missing"])
    t.add_argument("--max-categories", type=int, default=DEFAULTS["max_categories"])
    t.add_argument("--ablation", action="store_true",
                   help="also run each stimulus condition alone (3 extra runs per device case)")
    t.add_argument("--skip-mlp", action="store_true", help="skip the (slowest) software MLP reference")
    t.add_argument("--no-plots", action="store_true")
    p = sub.add_parser("predict", help="predict new patients with a saved model")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="predictions.csv")
    args = ap.parse_args(argv)
    try:  # never crash on consoles that cannot print symbols such as the subscript 2 / micro sign
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    if args.cmd == "train":
        dd = Path(args.device_dir)
        if not dd.exists() and (Path(__file__).resolve().parent / dd).exists():
            args.device_dir = str(Path(__file__).resolve().parent / dd)   # run from any folder
        if args.class_weight == "none":
            args.class_weight = None
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    # Run through the importable module (not __main__) so that saved .joblib models
    # can later be loaded anywhere with:  import alzheimer_device_model; joblib.load(...)
    import importlib
    _here = Path(__file__).resolve()
    sys.path.insert(0, str(_here.parent))
    try:
        _module = importlib.import_module(_here.stem)
    except Exception:  # unusual file name - still works, only notebook loading needs this file
        _module = None
    (_module.main if _module is not None else main)()

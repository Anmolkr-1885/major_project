#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alzheimer's disease stage classification from brain-MRI slices
================================================================
Model chosen from the 4 reference papers  ->  ConXNet-style CNN (Ref. 3)
  4 x [Conv3x3 -> ReLU -> BatchNorm -> MaxPool]  ->  GAP  ->  Dense  ->  softmax
  Adam (lr 1e-3), batch 32, Keras/TensorFlow  (same training recipe as Ref. 3),
  + augmentation, dropout, class weights (Refs. 2-3: small / imbalanced data),
  + Grad-CAM heat-maps (Ref. 3: heat-map interpretation),
  + metrics of Ref. 2 (accuracy, precision, recall/sensitivity, F1, AUC).
Reference baselines trained on the same split:
  ANN / MLP on pixels (Refs. 2-3), PCA + SVM (Ref. 1), CNN features + kernel SVM (Ref. 2).
Optional: VGG16 transfer learning (Ref. 2)  -> needs ImageNet weights (internet).

IMPORTANT - patient-level evaluation
------------------------------------
Datasets of this kind (e.g. 6400 JPG slices in 4 class folders) contain many
consecutive slices of the SAME patient.  A random image split puts slices of
one patient in train AND test, so a model can "recognise the patient" and reach
~99 % without learning dementia.  This script therefore (default --split subject)
reconstructs the patients from slice similarity (or from file names with
--group_regex) and keeps every patient entirely in train, val OR test.
Use --split random only to reproduce the (leaky) numbers found in many papers.

USAGE
-----
  python alzheimer_mri_cnn.py inspect --data_dir DATASET_FOLDER
  python alzheimer_mri_cnn.py train   --data_dir DATASET_FOLDER               # patient-level split
  python alzheimer_mri_cnn.py train   --data_dir DATASET_FOLDER --split random --out results_random
  python alzheimer_mri_cnn.py train   --data_dir DATASET_FOLDER --cv 5         # 5-fold patient-level CV
  python alzheimer_mri_cnn.py predict --model results_subject/model.keras --images NEW_FOLDER
DATASET_FOLDER = the unzipped dataset; one sub-folder per class (nested train/test folders are fine).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import warnings  # noqa: E402
warnings.filterwarnings("ignore", message=".*shuffle=True.*")
warnings.filterwarnings("ignore", message=".*structure of `inputs`.*")

import numpy as np
import pandas as pd
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_WORDS = {"train", "training", "test", "testing", "val", "valid", "validation"}
SEVERITY = [("non", 0), ("verymild", 1), ("mild", 2), ("moderate", 3), ("severe", 4)]

# ----------------------------------------------------------------------------
# 1. Dataset loading
# ----------------------------------------------------------------------------
def severity_key(name):
    n = re.sub(r"[^a-z]", "", name.lower())
    for key, rank in SEVERITY:
        if n.startswith(key) or (key == "verymild" and "verymild" in n):
            return (rank, name)
    return (99, name)


def discover(data_dir):
    """All images below data_dir; class = parent folder name (train/test folders are skipped)."""
    root = Path(data_dir)
    if not root.exists():
        sys.exit(f"[error] data_dir not found: {root}")
    rows, skipped = [], set()
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXT or any(part.startswith(("__MACOSX", ".")) for part in p.parts):
            continue
        rel = p.relative_to(root).parts[:-1]
        aug = [d for d in rel if "augment" in d.lower()]
        if aug:   # pre-augmented copies of the SAME patients -> would leak into test; never used
            skipped.add(aug[0])
            continue
        cls = p.parent.name
        orig = p.parent.parent.name.lower() if p.parent.parent.name.lower() in SPLIT_WORDS else ""
        rows.append({"path": str(p), "label": cls, "orig_split": orig})
    for d in sorted(skipped):
        print(f"[info] skipped folder '{d}': pre-augmented copies of the same patients (leak into test). "
              "On-the-fly augmentation is used for training instead.")
    if not rows:
        sys.exit(f"[error] no images found below {root}")
    df = pd.DataFrame(rows)
    classes = sorted(df.label.unique(), key=severity_key)
    if len(classes) < 2:
        sys.exit(f"[error] need >=2 class folders, found {classes}")
    return df, classes


def load_images(paths, size, cache=None):
    if cache and Path(cache).exists():
        d = np.load(cache, allow_pickle=True)
        if list(d["paths"]) == list(paths) and int(d["size"]) == size:
            return d["X"]
    X = np.empty((len(paths), size, size), dtype=np.uint8)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("L")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        X[i] = np.asarray(im, dtype=np.uint8)
    if cache:
        np.savez_compressed(cache, X=X, paths=np.array(paths), size=size)
    return X


# ----------------------------------------------------------------------------
# 2. Patient (subject) reconstruction  +  leakage check
# ----------------------------------------------------------------------------
def _embed(X, side=64):
    Z = np.stack([np.asarray(Image.fromarray(x).resize((side, side), Image.BILINEAR), np.float32).ravel()
                  for x in X])
    Z -= Z.mean(1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    return Z


def nearest_neighbours(Z, k=2, chunk=1024):
    n = len(Z)
    idx = np.empty((n, k), dtype=np.int64)
    sim = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, chunk):
        S = Z[s:s + chunk] @ Z.T
        S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -np.inf
        top = np.argpartition(-S, k, axis=1)[:, :k]
        ts = np.take_along_axis(S, top, 1)
        order = np.argsort(-ts, axis=1)
        idx[s:s + chunk] = np.take_along_axis(top, order, 1)
        sim[s:s + chunk] = np.take_along_axis(ts, order, 1)
    return idx, sim


def infer_subjects(X, labels, threshold=0.98, group_regex=None, paths=None):
    """Group slices that belong to the same scan (patient).
    Links every slice to its 2 most similar slices if their correlation > threshold
    (consecutive slices of one brain are almost identical, different brains are not)
    and takes connected components.  Returns group ids + diagnostics."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if group_regex:
        ids = []
        for p in paths:
            m = re.search(group_regex, Path(p).name) or re.search(group_regex, str(p))
            ids.append(m.group(1) if m and m.groups() else (m.group(0) if m else Path(p).stem))
        codes, groups = np.unique(np.array(ids), return_inverse=True)
        return groups, {"method": f"regex {group_regex}", "n_groups": int(len(codes))}
    Z = _embed(X)
    idx, sim = nearest_neighbours(Z, k=2)
    labels = np.asarray(labels)
    nn_acc = float((labels[idx[:, 0]] == labels).mean())
    rows, cols = [], []
    for k in range(2):
        m = sim[:, k] > threshold
        rows += list(np.where(m)[0])
        cols += list(idx[m, k])
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(X), len(X)))
    n, groups = connected_components(A, directed=False)
    sizes = np.bincount(groups)
    mixed = int(sum(len(set(labels[groups == g])) > 1 for g in range(n)))
    info = {"method": f"slice-similarity graph (corr > {threshold})", "n_groups": int(n),
            "group_size_min": int(sizes.min()), "group_size_median": float(np.median(sizes)),
            "group_size_max": int(sizes.max()), "groups_with_mixed_labels": mixed,
            "nn_same_label_fraction": nn_acc,
            "nn_corr_median": float(np.median(sim[:, 0]))}
    return groups, info


# ----------------------------------------------------------------------------
# 3. Splits (patient-level or random)
# ----------------------------------------------------------------------------
def group_label(y, groups):
    gl = {}
    for g in np.unique(groups):
        vals, cnt = np.unique(y[groups == g], return_counts=True)
        gl[g] = vals[np.argmax(cnt)]
    return gl


def subject_split(y, groups, test_frac, val_frac, seed):
    """Per class: shuffle its patients, give ~test_frac to test (>=1 if the class has >=2
    patients), ~val_frac to validation (>=1 if >=3 patients), the rest to train."""
    rng = np.random.default_rng(seed)
    gl = group_label(y, groups)
    split_of = {}
    for c in np.unique(y):
        gs = np.array([g for g, l in gl.items() if l == c])
        rng.shuffle(gs)
        n = len(gs)
        n_test = max(1, int(round(n * test_frac))) if (n >= 2 and test_frac > 0) else 0
        n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
        n_test = min(n_test, n - 1) if n >= 2 else 0
        n_test = n_test if test_frac > 0 else 0
        n_val = min(n_val, n - n_test - 1) if n - n_test >= 2 else 0
        for i, g in enumerate(gs):
            split_of[g] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return np.array([split_of[g] for g in groups])


def subject_cv_splits(y, groups, k, val_frac, seed):
    rng = np.random.default_rng(seed)
    gl = group_label(y, groups)
    fold_of = {}
    for c in np.unique(y):
        gs = np.array([g for g, l in gl.items() if l == c])
        rng.shuffle(gs)
        for i, g in enumerate(gs):
            fold_of[g] = i % k
    folds = np.array([fold_of[g] for g in groups])
    out = []
    for f in range(k):
        split = np.where(folds == f, "test", "train").astype(object)
        tr_mask = split == "train"
        sub = subject_split(y[tr_mask], groups[tr_mask], test_frac=0.0, val_frac=val_frac, seed=seed + f)
        split[np.where(tr_mask)[0][sub == "val"]] = "val"
        out.append(split.astype(str))
    return out


def random_split(y, test_frac, val_frac, seed):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tv, te = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    tr, va = train_test_split(tv, test_size=val_frac / (1 - test_frac), stratify=y[tv], random_state=seed)
    split = np.empty(len(y), dtype=object)
    split[tr], split[va], split[te] = "train", "val", "test"
    return split.astype(str)


# ----------------------------------------------------------------------------
# 4. Models
# ----------------------------------------------------------------------------
def _augment(layers, keras):
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.04),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomZoom(0.08),
        layers.RandomContrast(0.1),
    ], name="augmentation")


def build_model(arch, img, n_classes, augment=True, weights="imagenet"):
    import keras
    from keras import layers
    inp = keras.Input((img, img, 1), name="mri")
    x = _augment(layers, keras)(inp) if augment else inp
    if arch == "conxnet":                                   # selected model (Ref. 3 blocks)
        for i, f in enumerate((32, 64, 128, 256), 1):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
            x = layers.ReLU(name=f"relu{i}")(x)
            x = layers.BatchNormalization(name=f"bn{i}")(x)
            x = layers.MaxPooling2D(name=f"pool{i}")(x)
        x = layers.GlobalAveragePooling2D(name="features")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation="relu", name="fc1")(x)
        x = layers.Dropout(0.3)(x)
    elif arch == "mlp":                                     # ANN baseline (Refs. 2-3)
        x = layers.Resizing(64, 64)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu", name="fc0")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation="relu", name="fc1")(x)
        x = layers.Dropout(0.3)(x)
    elif arch == "vgg16":                                   # transfer learning (Ref. 2)
        x = layers.Concatenate()([x, x, x])
        x = layers.Rescaling(255.0)(x)
        x = keras.applications.vgg16.preprocess_input(x)
        base = keras.applications.VGG16(include_top=False, weights=None if weights == "none" else weights,
                                        input_shape=(img, img, 3))
        base.trainable = False
        x = base(x)
        x = layers.GlobalAveragePooling2D(name="features")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation="relu", name="fc1")(x)
        x = layers.Dropout(0.3)(x)
    else:
        raise ValueError(arch)
    out = layers.Dense(n_classes, activation="softmax", name="classifier")(x)
    return keras.Model(inp, out, name=arch)


def make_ds(X, y, batch, shuffle, seed=0):
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((X[..., None], y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch).map(lambda a, b: (tf.cast(a, tf.float32) / 255.0, b),
                             num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def train_keras(arch, Xtr, ytr, Xva, yva, n_classes, args, out_dir, tag):
    import keras
    keras.utils.set_random_seed(args.seed)
    model = build_model(arch, args.img_size, n_classes, augment=not args.no_augment, weights=args.weights)
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    counts = np.bincount(ytr, minlength=n_classes)
    cw = {i: (len(ytr) / (n_classes * c) if c > 0 else 0.0) for i, c in enumerate(counts)}
    has_val = len(yva) > 0
    mon = "val_loss" if has_val else "loss"
    cbs = [keras.callbacks.EarlyStopping(monitor=mon, patience=args.patience, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor=mon, factor=0.5, patience=max(3, args.patience // 3),
                                             min_lr=1e-5),
           keras.callbacks.CSVLogger(str(out_dir / f"history_{tag}.csv"))]
    t0 = time.time()
    hist = model.fit(make_ds(Xtr, ytr, args.batch, True, args.seed),
                     validation_data=make_ds(Xva, yva, args.batch, False) if has_val else None,
                     epochs=args.epochs, class_weight=cw if not args.no_class_weight else None,
                     callbacks=cbs, verbose=2 if args.verbose else 0)
    return model, pd.DataFrame(hist.history), time.time() - t0


def predict_proba(model, X, batch=64):
    return model.predict(make_ds(X, np.zeros(len(X), np.int64), batch, False), verbose=0)


def feature_extractor(model):
    import keras
    return keras.Model(model.input, model.get_layer("features").output)


# ----------------------------------------------------------------------------
# 5. Metrics
# ----------------------------------------------------------------------------
def metrics_from_proba(y, proba, classes):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    pred = proba.argmax(1)
    present = np.unique(y)
    m = {"n": int(len(y)), "accuracy": accuracy_score(y, pred),
         "balanced_accuracy": balanced_accuracy_score(y, pred),
         "precision_macro": precision_score(y, pred, labels=present, average="macro", zero_division=0),
         "recall_macro": recall_score(y, pred, labels=present, average="macro", zero_division=0),
         "f1_macro": f1_score(y, pred, labels=present, average="macro", zero_division=0)}
    try:
        if len(present) == proba.shape[1]:
            m["roc_auc_ovr"] = roc_auc_score(y, proba, multi_class="ovr", average="macro")
        elif len(present) > 1:
            p = proba[:, present] / proba[:, present].sum(1, keepdims=True)
            yy = np.searchsorted(present, y)
            m["roc_auc_ovr"] = roc_auc_score(yy, p, multi_class="ovr", average="macro") if len(present) > 2 \
                else roc_auc_score(yy, p[:, 1])
    except ValueError:
        m["roc_auc_ovr"] = float("nan")
    # clinically useful binary view: any dementia vs non-demented (class 0 = least severe)
    if proba.shape[1] > 2 and severity_key(classes[0])[0] == 0:
        yb, pb = (y > 0).astype(int), 1 - proba[:, 0]
        m["binary_accuracy_demented_vs_non"] = accuracy_score(yb, (pb >= 0.5).astype(int))
        m["binary_sensitivity"] = recall_score(yb, (pb >= 0.5).astype(int), zero_division=0)
        m["binary_specificity"] = recall_score(1 - yb, (pb < 0.5).astype(int), zero_division=0)
        try:
            m["binary_auc"] = roc_auc_score(yb, pb)
        except ValueError:
            m["binary_auc"] = float("nan")
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}


def patient_level(y, proba, groups):
    """Diagnose a patient = mean probability over all of his/her test slices."""
    df = pd.DataFrame(proba)
    df["g"], df["y"] = groups, y
    agg = df.groupby("g").mean(numeric_only=True)
    yt = df.groupby("g")["y"].agg(lambda s: s.mode().iloc[0]).loc[agg.index].to_numpy().astype(int)
    pr = agg.drop(columns=["y"]).to_numpy()
    return yt, pr


# ----------------------------------------------------------------------------
# 6. Device-aware option (IGZO synaptic transistor weights for the dense layers)
# ----------------------------------------------------------------------------
def device_levels(device_dir, n_levels=32, vg=(-1.0, 3.0), vds=1.0):
    from scipy.interpolate import PchipInterpolator
    tr = pd.read_csv(Path(device_dir) / "transfer_curves.csv")
    out = {}
    for case, t in tr[np.isclose(tr.Vds_V, vds)].groupby("case"):
        t = t.groupby("Vg_V", as_index=False).Id_A.mean()
        v = np.linspace(vg[0], vg[1], n_levels)
        out[case] = 10 ** PchipInterpolator(t.Vg_V, np.log10(np.abs(t.Id_A)))(v) / vds
    return out


def program_weights(W, g, noise=0.0, rng=None):
    """Differential pair w = s (G+ - G-)/(Gmax-Gmin); |w|/s -> nearest device level (per output neuron)."""
    g = np.asarray(g, float)
    gmin, span = g.min(), g.max() - g.min()
    gn = (g - gmin) / span
    rng = rng or np.random.default_rng(0)
    W = np.asarray(W, float)
    out = np.zeros_like(W)
    for j in range(W.shape[1]):                      # Keras Dense kernel: (inputs, outputs)
        col = W[:, j]
        s = np.abs(col).max()
        if s == 0:
            continue
        sel = g[np.abs(np.abs(col)[:, None] / s - gn[None, :]).argmin(1)]
        gp, gm = np.where(col >= 0, sel, gmin), np.where(col >= 0, gmin, sel)
        if noise > 0:
            gp = gp * (1 + noise * rng.standard_normal(gp.shape))
            gm = gm * (1 + noise * rng.standard_normal(gm.shape))
        out[:, j] = s * (gp - gm) / span
    return out


def device_eval(model, Xte, yte, classes, device_dir, n_levels, noise_list=(0.0, 0.05)):
    rows = []
    dense = [l for l in model.layers if l.name in ("fc1", "classifier")]
    orig = {l.name: [w.copy() for w in l.get_weights()] for l in dense}
    for case, g in device_levels(device_dir, n_levels).items():
        for noise in noise_list:
            rng = np.random.default_rng(1)
            for l in dense:
                k, b = orig[l.name]
                l.set_weights([program_weights(k, g, noise, rng).astype(np.float32), b])
            m = metrics_from_proba(yte, predict_proba(model, Xte), classes)
            rows.append({"case": case, "levels": n_levels, "noise": noise, **m})
    for l in dense:
        l.set_weights(orig[l.name])
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 7. Figures (light surface, fixed colours, text never in series colour)
# ----------------------------------------------------------------------------
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.labelcolor": INK2, "font.size": 10, "lines.linewidth": 2})
    return plt


def fig_dataset(df, classes, groups, split, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    img_counts = df.label.value_counts().reindex(classes)
    subj = pd.Series(groups).groupby(df.label.values).nunique().reindex(classes)
    for ax, s, t in [(axes[0], img_counts, "Images per class"), (axes[1], subj, "Patients per class (reconstructed)")]:
        ax.barh(range(len(classes))[::-1], s.values, height=0.5, color=SERIES[0])
        for yy, v in zip(range(len(classes))[::-1], s.values):
            ax.text(v, yy, f"  {v}", va="center", color=INK, fontsize=9)
        ax.set_yticks(range(len(classes))[::-1])
        ax.set_yticklabels(classes)
        ax.set_title(t, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, s.max() * 1.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_history(hist, path, title):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ep = np.arange(1, len(hist) + 1)
    for ax, key, lab in [(axes[0], "loss", "Loss (cross-entropy)"), (axes[1], "accuracy", "Accuracy")]:
        ax.plot(ep, hist[key], color=SERIES[0], label="train")
        if f"val_{key}" in hist:
            ax.plot(ep, hist[f"val_{key}"], color=SERIES[1], label="validation")
        ax.set_xlabel("epoch")
        ax.set_title(lab, loc="left")
        ax.legend()
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_confusion(y, pred, classes, path, title):
    from sklearn.metrics import confusion_matrix
    from matplotlib.colors import LinearSegmentedColormap
    plt = _plt()
    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("b", ["#f4f8fe", "#86b6ef", "#2a78d6", "#104281"])
    fig, ax = plt.subplots(figsize=(1.4 * len(classes) + 2.4, 1.2 * len(classes) + 1.8))
    ax.imshow(rown, cmap=cmap, vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}\n{rown[i, j]:.0%}" if cm[i].sum() else "-", ha="center", va="center", fontsize=9,
                color="white" if rown[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig_roc(y, proba, classes, path, title):
    from sklearn.metrics import roc_auc_score, roc_curve
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
    for k, c in enumerate(classes):
        yk = (y == k).astype(int)
        if 0 < yk.sum() < len(yk):
            f, t, _ = roc_curve(yk, proba[:, k])
            ax.plot(f, t, color=SERIES[k % len(SERIES)], label=f"{c} (AUC {roc_auc_score(yk, proba[:, k]):.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def gradcam(model, x, class_idx, conv_name="bn4"):
    import tensorflow as tf
    import keras
    grad_model = keras.Model(model.input, [model.get_layer(conv_name).output, model.output])
    with tf.GradientTape() as tape:
        conv, pred = grad_model(x[None, ..., None].astype("float32") / 255.0, training=False)
        loss = pred[:, class_idx]
    g = tape.gradient(loss, conv)[0]
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * tf.reduce_mean(g, axis=(0, 1)), axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    return np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize(x.shape[::-1], Image.BILINEAR)) / 255.0


def fig_gradcam(model, X, y, proba, classes, path, per_class=3):
    plt = _plt()
    rng = np.random.default_rng(0)
    rows = [k for k in range(len(classes)) if (y == k).any()]
    fig, axes = plt.subplots(len(rows), per_class * 2, figsize=(2.0 * per_class * 2, 2.1 * len(rows)),
                             squeeze=False)
    for r, k in enumerate(rows):
        idx = rng.choice(np.where(y == k)[0], size=min(per_class, (y == k).sum()), replace=False)
        for c in range(per_class):
            a0, a1 = axes[r, 2 * c], axes[r, 2 * c + 1]
            for a in (a0, a1):
                a.axis("off")
            if c >= len(idx):
                continue
            i = idx[c]
            cam = gradcam(model, X[i], int(proba[i].argmax()))
            a0.imshow(X[i], cmap="gray")
            a1.imshow(X[i], cmap="gray")
            a1.imshow(cam, cmap="jet", alpha=0.4)
            a0.set_title(f"true: {classes[k]}", fontsize=7, loc="left")
            a1.set_title(f"pred: {classes[int(proba[i].argmax())]} ({proba[i].max():.2f})", fontsize=7, loc="left")
    fig.suptitle("Grad-CAM: regions that drove the CNN decision (test patients)", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_models(tab, path, title, highlight="CNN (ConXNet, selected)"):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.5 * len(tab) + 1.8), sharey=True)
    ypos = np.arange(len(tab))[::-1]
    for ax, met, t in [(axes[0], "balanced_accuracy", "Balanced accuracy"), (axes[1], "f1_macro", "Macro F1")]:
        for yy, (_, r) in zip(ypos, tab.iterrows()):
            col = SERIES[0] if r.model == highlight else MUTED
            ax.plot([0, r[met]], [yy, yy], color=GRID, lw=1)
            ax.plot(r[met], yy, "o", ms=9, color=col, mec=SURFACE, mew=2)
            ax.text(r[met], yy + 0.22, f"{r[met]:.3f}", ha="center", va="bottom", color=INK, fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(-0.6, len(tab) - 0.3)
        ax.set_title(t, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_yticks(ypos)
        ax.set_yticklabels(tab.model)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 8. Baselines (fast, scikit-learn)
# ----------------------------------------------------------------------------
def svm_on(Ftr, ytr, Fva, yva, Fte, seed):
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    sc = StandardScaler().fit(Ftr)
    best, bestC = -1, 1.0
    for C in (0.1, 1, 10, 100):
        if len(yva) == 0:
            break
        m = SVC(C=C, kernel="rbf", class_weight="balanced", random_state=seed).fit(sc.transform(Ftr), ytr)
        s = (m.predict(sc.transform(Fva)) == yva).mean()
        if s > best:
            best, bestC = s, C
    m = SVC(C=bestC, kernel="rbf", class_weight="balanced", probability=True, random_state=seed)
    m.fit(sc.transform(Ftr), ytr)
    P = np.zeros((len(Fte), int(max(ytr.max(), 0)) + 1))
    P[:, m.classes_] = m.predict_proba(sc.transform(Fte))
    return P


def pca_svm(Xtr, ytr, Xva, yva, Xte, n_classes, seed):
    from sklearn.decomposition import PCA
    def small(A):
        return np.stack([np.asarray(Image.fromarray(a).resize((64, 64), Image.BILINEAR), np.float32).ravel()
                         for a in A]) / 255.0
    Ttr, Tva, Tte = small(Xtr), small(Xva), small(Xte)
    pca = PCA(n_components=min(100, len(Ttr) - 1), random_state=seed).fit(Ttr)
    P = svm_on(pca.transform(Ttr), ytr, pca.transform(Tva), yva, pca.transform(Tte), seed)
    return _pad(P, n_classes)


def _pad(P, n):
    if P.shape[1] == n:
        return P
    Q = np.zeros((len(P), n))
    Q[:, :P.shape[1]] = P
    return Q


# ----------------------------------------------------------------------------
# 9. Train / evaluate one split
# ----------------------------------------------------------------------------
def run_split(X, y, groups, split, classes, args, out_dir, tag="main", figures=True):
    import tensorflow as tf  # noqa: F401
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    print(f"  [{tag}] train {tr.sum()} slices / {len(np.unique(groups[tr]))} patients | "
          f"val {va.sum()} / {len(np.unique(groups[va]))} | test {te.sum()} / {len(np.unique(groups[te]))}")
    n = len(classes)
    results, probas = [], {}

    model, hist, secs = train_keras(args.arch, X[tr], y[tr], X[va], y[va], n, args, out_dir, tag)
    P = predict_proba(model, X[te])
    name = {"conxnet": "CNN (ConXNet, selected)", "mlp": "ANN / MLP", "vgg16": "VGG16 transfer"}[args.arch]
    probas[name] = P
    results.append({"model": name, "epochs": len(hist), "train_seconds": round(secs, 1),
                    **metrics_from_proba(y[te], P, classes)})
    if args.arch == "conxnet" and not args.no_baselines:
        # CNN features + kernel SVM (Ref. 2)
        fx = feature_extractor(model)

        def feat(A):
            return fx.predict(make_ds(A, np.zeros(len(A), np.int64), 64, False), verbose=0)
        Ps = _pad(svm_on(feat(X[tr]), y[tr], feat(X[va]), y[va], feat(X[te]), args.seed), n)
        probas["CNN features + SVM"] = Ps
        results.append({"model": "CNN features + SVM", **metrics_from_proba(y[te], Ps, classes)})
        # PCA + SVM (Ref. 1) and ANN/MLP (Refs. 2-3)
        Pp = pca_svm(X[tr], y[tr], X[va], y[va], X[te], n, args.seed)
        probas["PCA + SVM"] = Pp
        results.append({"model": "PCA + SVM", **metrics_from_proba(y[te], Pp, classes)})
        a2 = argparse.Namespace(**{**vars(args), "arch": "mlp"})
        mlp, hmlp, smlp = train_keras("mlp", X[tr], y[tr], X[va], y[va], n, a2, out_dir, f"{tag}_mlp")
        Pm = predict_proba(mlp, X[te])
        probas["ANN / MLP"] = Pm
        results.append({"model": "ANN / MLP", "epochs": len(hmlp), "train_seconds": round(smlp, 1),
                        **metrics_from_proba(y[te], Pm, classes)})
    # patient-level diagnosis for the main model
    yt_p, pr_p = patient_level(y[te], P, groups[te])
    pat = metrics_from_proba(yt_p, pr_p, classes)
    dev = None
    if args.device_dir and Path(args.device_dir, "transfer_curves.csv").exists():
        dev = device_eval(model, X[te], y[te], classes, args.device_dir, args.n_levels)
    if figures:
        fd = out_dir / "figures"
        fd.mkdir(exist_ok=True)
        fig_history(hist, fd / f"training_curves_{tag}.png", f"Training history - {name}")
        fig_confusion(y[te], P.argmax(1), classes, fd / f"confusion_matrix_{tag}.png",
                      f"{name}: test slices (n={te.sum()})")
        fig_confusion(yt_p, pr_p.argmax(1), classes, fd / f"confusion_matrix_patients_{tag}.png",
                      f"{name}: test patients (n={len(yt_p)})")
        fig_roc(y[te], P, classes, fd / f"roc_{tag}.png", f"ROC one-vs-rest - {name}")
        if args.arch == "conxnet":
            fig_gradcam(model, X[te], y[te], P, classes, fd / f"gradcam_{tag}.png")
        if len(results) > 1:
            fig_models(pd.DataFrame(results), fd / f"model_comparison_{tag}.png",
                       f"Test performance ({args.split} split) - selected CNN vs reference baselines")
    return model, pd.DataFrame(results), pat, probas, dev, hist


# ----------------------------------------------------------------------------
# 10. Commands
# ----------------------------------------------------------------------------
def prepare(args):
    df, classes = discover(args.data_dir)
    y = df.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    cache = Path(args.out) / f"images_{args.img_size}.npz" if args.cache else None
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
    X = load_images(df.path.tolist(), args.img_size, cache)
    groups, ginfo = infer_subjects(X, y, args.group_threshold, args.group_regex, df.path.tolist())
    return df, classes, X, y, groups, ginfo


def cmd_inspect(args):
    args.cache, args.out = False, "."
    df, classes, X, y, groups, gi = prepare(args)
    print(f"images: {len(df)}   classes: {classes}")
    print(df.label.value_counts().reindex(classes).to_string())
    print("\npatient reconstruction:", json.dumps(gi, indent=2))
    if "nn_same_label_fraction" in gi:
        print(f"\n-> {gi['nn_same_label_fraction']:.1%} of slices have their most similar slice in the SAME class.")
        print("   Near 100 % means a random image split leaks patients into the test set; use --split subject.")
    print("\npatients per class:")
    print(pd.Series(groups).groupby(df.label.values).nunique().reindex(classes).to_string())


def cmd_train(args):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df, classes, X, y, groups, ginfo = prepare(args)
    if args.merge_moderate:
        mod = [i for i, c in enumerate(classes) if "moderate" in c.lower()]
        mild = [i for i, c in enumerate(classes) if "mild" in c.lower() and "very" not in c.lower()]
        if mod and mild:
            y = np.where(y == mod[0], mild[0], y)
            classes = [c if i != mild[0] else f"{c}+{classes[mod[0]]}" for i, c in enumerate(classes)]
            keep = [i for i in range(len(classes)) if i != mod[0]]
            remap = {old: new for new, old in enumerate(keep)}
            y = np.array([remap[v] for v in y])
            classes = [classes[i] for i in keep]
    print("=" * 78)
    print(f"Dataset: {len(df)} images, classes: " + ", ".join(f"{c}={int((y == i).sum())}" for i, c in enumerate(classes)))
    print(f"Patients: {ginfo}")
    print(f"Split: {args.split}  |  model: {args.arch}  |  image {args.img_size}px  |  seed {args.seed}")
    print("=" * 78)
    cfg = {**vars(args), "classes": classes, "patients": ginfo, "n_images": int(len(df))}

    if args.cv and args.cv > 1:
        if args.split != "subject":
            sys.exit("[error] --cv is implemented for patient-level folds (--split subject)")
        folds = subject_cv_splits(y, groups, args.cv, args.val_frac, args.seed)
        all_rows, oof, pats = [], np.zeros((len(y), len(classes))), []
        for f, split in enumerate(folds):
            model, res, pat, probas, dev, hist = run_split(X, y, groups, split, classes, args, out_dir,
                                                           tag=f"fold{f + 1}", figures=(f == 0))
            res["fold"] = f + 1
            all_rows.append(res)
            te = split == "test"
            oof[te] = probas[res.model.iloc[0]]
            pats.append({"fold": f + 1, **pat})
        res = pd.concat(all_rows)
        res.to_csv(out_dir / "cv_metrics_per_fold.csv", index=False)
        summ = res.groupby("model").agg(["mean", "std"]).drop(columns=["fold"], errors="ignore")
        summ.to_csv(out_dir / "cv_metrics_summary.csv")
        pooled = metrics_from_proba(y, oof, classes)
        pd.DataFrame(pats).to_csv(out_dir / "cv_patient_level.csv", index=False)
        json.dump({"pooled_out_of_fold": pooled, "config": cfg}, open(out_dir / "metrics.json", "w"),
                  indent=2, default=str)
        fig_confusion(y, oof.argmax(1), classes, out_dir / "figures" / "confusion_matrix_cv_pooled.png",
                      f"{args.cv}-fold patient-level CV, pooled test slices")
        print(summ.round(3).to_string())
        print("pooled out-of-fold:", json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                                for k, v in pooled.items()}))
        print(f"\nDone in {time.time() - t0:.0f} s -> {out_dir.resolve()}")
        return

    if args.split == "subject":
        split = subject_split(y, groups, args.test_frac, args.val_frac, args.seed)
    else:
        split = random_split(y, args.test_frac, args.val_frac, args.seed)
    leak = len(set(groups[split == "train"]) & set(groups[split == "test"]))
    print(f"patients shared by train and test: {leak}")
    pd.DataFrame({"path": df.path, "label": [classes[i] for i in y], "patient_group": groups, "split": split}) \
        .to_csv(out_dir / "split_assignment.csv", index=False)
    fd = out_dir / "figures"
    fd.mkdir(exist_ok=True)
    fig_dataset(df.assign(label=[classes[i] for i in y]), classes, groups, split, fd / "dataset_overview.png")

    model, res, pat, probas, dev, hist = run_split(X, y, groups, split, classes, args, out_dir)
    model.save(out_dir / "model.keras")
    json.dump(classes, open(out_dir / "class_names.json", "w"))
    te = split == "test"
    P = probas[res.model.iloc[0]]
    from sklearn.metrics import classification_report
    rep = classification_report(y[te], P.argmax(1), labels=range(len(classes)), target_names=classes,
                                zero_division=0, digits=3)
    (out_dir / "classification_report.txt").write_text(rep)
    pred_df = pd.DataFrame({"path": df.path[te].values, "true": [classes[i] for i in y[te]],
                            "pred": [classes[i] for i in P.argmax(1)], "patient_group": groups[te]})
    for k, c in enumerate(classes):
        pred_df[f"P({c})"] = P[:, k]
    pred_df.to_csv(out_dir / "predictions_test.csv", index=False, float_format="%.4f")
    res.to_csv(out_dir / "metrics_models.csv", index=False, float_format="%.4f")
    if dev is not None:
        dev.to_csv(out_dir / "device_mapping.csv", index=False, float_format="%.4f")
    json.dump({"slice_level": res.iloc[0].to_dict(), "patient_level": pat, "patients_shared_train_test": leak,
               "config": cfg}, open(out_dir / "metrics.json", "w"), indent=2, default=str)

    show = res[["model", "accuracy", "balanced_accuracy", "f1_macro", "roc_auc_ovr"] +
               [c for c in ["binary_accuracy_demented_vs_non", "binary_auc"] if c in res]]
    print("\nTEST (slice level):\n" + show.round(4).to_string(index=False))
    print("\nTEST (patient level, main model):", json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                                          for k, v in pat.items()}))
    print("\n" + rep)
    if dev is not None:
        print("Device-mapped dense layers (IGZO levels):\n" +
              dev[["case", "levels", "noise", "accuracy", "balanced_accuracy"]].round(4).to_string(index=False))
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(f"# Alzheimer MRI classification - {args.split} split\n\n")
        f.write(f"- images {len(df)}, classes {classes}\n- patients: {ginfo}\n")
        f.write(f"- patients shared by train and test: {leak}\n- model: {args.arch}, {args.img_size}px, "
                f"seed {args.seed}\n\n## Test, slice level\n\n")
        f.write("| " + " | ".join(show.columns) + " |\n|" + "---|" * len(show.columns) + "\n")
        for _, r in show.iterrows():
            f.write("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |\n")
        f.write("\n## Test, patient level (main model)\n\n" + json.dumps(pat, indent=2) + "\n\n```\n" + rep + "```\n")
    print(f"\nDone in {time.time() - t0:.0f} s -> {out_dir.resolve()}")


def cmd_predict(args):
    import keras
    model = keras.models.load_model(args.model)
    classes = json.load(open(Path(args.model).parent / "class_names.json"))
    size = model.input_shape[1]
    p = Path(args.images)
    files = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXT)
    X = load_images([str(f) for f in files], size)
    P = predict_proba(model, X)
    out = pd.DataFrame({"file": [str(f) for f in files], "prediction": [classes[i] for i in P.argmax(1)],
                        "confidence": P.max(1)})
    for k, c in enumerate(classes):
        out[f"P({c})"] = P[:, k]
    out.to_csv(args.out, index=False, float_format="%.4f")
    print(out.head(20).to_string(index=False))
    print(f"... {len(out)} predictions -> {args.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "train"):
        s = sub.add_parser(name)
        s.add_argument("--data_dir", required=True)
        s.add_argument("--img_size", type=int, default=128)
        s.add_argument("--group_threshold", type=float, default=0.98,
                       help="slice correlation that links two slices to the same patient")
        s.add_argument("--group_regex", default=None,
                       help="regex whose first group is the patient id in the file name (if available)")
        if name == "train":
            s.add_argument("--out", default="results")
            s.add_argument("--split", choices=["subject", "random"], default="subject")
            s.add_argument("--cv", type=int, default=0, help="k-fold patient-level CV instead of one split")
            s.add_argument("--test_frac", type=float, default=0.2)
            s.add_argument("--val_frac", type=float, default=0.15)
            s.add_argument("--arch", choices=["conxnet", "mlp", "vgg16"], default="conxnet")
            s.add_argument("--weights", default="imagenet", help="VGG16 weights: imagenet | none")
            s.add_argument("--epochs", type=int, default=40)
            s.add_argument("--batch", type=int, default=32)
            s.add_argument("--lr", type=float, default=1e-3)
            s.add_argument("--patience", type=int, default=10)
            s.add_argument("--seed", type=int, default=42)
            s.add_argument("--merge_moderate", action="store_true",
                           help="merge Moderate into Mild (only 2 patients are Moderate in the usual dataset)")
            s.add_argument("--no_augment", action="store_true")
            s.add_argument("--no_class_weight", action="store_true")
            s.add_argument("--no_baselines", action="store_true")
            s.add_argument("--device_dir", default=None,
                           help="folder with transfer_curves.csv -> test with IGZO-device dense weights")
            s.add_argument("--n_levels", type=int, default=32)
            s.add_argument("--cache", action="store_true", help="cache decoded images in the output folder")
            s.add_argument("--verbose", action="store_true")
    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", default="predictions.csv")
    args = ap.parse_args(argv)
    {"inspect": cmd_inspect, "train": cmd_train, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()

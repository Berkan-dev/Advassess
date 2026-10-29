"""Test-data loading shared by every numeric model type (no PyTorch or TensorFlow needed)."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def pick(n_total: int, max_samples: int, seed: int) -> np.ndarray:
    """Random subset of indices, so a capped run is still representative."""
    n = min(max_samples, n_total)
    return np.random.default_rng(seed).permutation(n_total)[:n]


def load_npz(path: str | Path, expected_shape: tuple | None, max_samples: int, seed: int):
    """x_test and y_test arrays from a .npz file. y may be class indices or one-hot."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        npz = np.load(path)
    except Exception:
        raise ValueError(f"{path.name} is not a NumPy .npz file. Create one with "
                         "np.savez('data.npz', x_test=..., y_test=...)") from None
    with npz:
        if "x_test" not in npz or "y_test" not in npz:
            raise KeyError("The .npz file must contain arrays named x_test and y_test")
        x, y = npz["x_test"], npz["y_test"]
    if y.ndim > 1:
        y = y.argmax(axis=1)
    x = x.astype(np.float32)
    if expected_shape is not None and tuple(x.shape[1:]) != tuple(expected_shape):
        raise ValueError(f"Data samples have shape {tuple(x.shape[1:])} but target.input_shape is {tuple(expected_shape)}")
    idx = pick(len(x), max_samples, seed)
    return x[idx], y[idx]


def _number_or_text(values: list[str]) -> np.ndarray:
    try:
        arr = np.asarray([float(v) for v in values])
        return arr.astype(np.int64) if np.all(arr == np.round(arr)) else arr
    except ValueError:
        return np.asarray(values)


def load_numeric_csv(path: str | Path, label_column: str, feature_columns: list[str] | None,
                     max_samples: int, seed: int):
    """Numeric features plus a label column. Returns x (N, F) and the raw labels."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        if label_column not in cols:
            raise KeyError(f"Column '{label_column}' not found in {path.name}. Columns are: {cols}. "
                           "Set data.label_column in the config.")
        feats = feature_columns or [c for c in cols if c != label_column]
        missing = [c for c in feats if c not in cols]
        if missing:
            raise KeyError(f"feature column(s) not found in {path.name}: {missing}")
        rows = list(reader)
    try:
        x = np.asarray([[float(r[c]) for c in feats] for r in rows], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"A feature value in {path.name} is not a number ({exc}). Use data.feature_columns to "
                         "choose numeric columns only.") from None
    labels = _number_or_text([r[label_column].strip() for r in rows])
    idx = pick(len(x), max_samples, seed)
    return x[idx], labels[idx]


def infer_clip_values(x: np.ndarray) -> tuple[float, float]:
    """Valid input range when the config doesn't say: snap to 0..1 or 0..255 when the data looks like it."""
    lo, hi = float(x.min()), float(x.max())
    if lo >= 0.0 and hi <= 1.0:
        return 0.0, 1.0
    if lo >= 0.0 and hi <= 255.0 and hi > 1.0 and np.allclose(x, np.round(x)):
        return 0.0, 255.0
    return lo, hi


def load_numeric(cfg: dict):
    """(x, y) for numeric models from data.source npz or csv; y as integer class numbers."""
    d, seed = cfg["data"], cfg["run"]["seed"]
    shape = cfg["target"].get("input_shape")
    if d["source"] == "csv":
        x, labels = load_numeric_csv(d["path"], d["label_column"], d.get("feature_columns"), d["max_samples"], seed)
        if labels.dtype.kind not in "iu":
            raise ValueError("For PyTorch and Keras models the label column must hold class numbers 0, 1, 2 ...")
        if shape is not None and tuple(x.shape[1:]) != tuple(shape):
            raise ValueError(f"The csv has {x.shape[1]} features but target.input_shape is {tuple(shape)}")
        return x, labels.astype(np.int64)
    if d["source"] == "npz":
        return load_npz(d["path"], shape, d["max_samples"], seed)
    raise ValueError("data.source must be npz or csv for this model type")

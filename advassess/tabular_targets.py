"""scikit-learn models on numeric (tabular) data, attacked through ART."""
from __future__ import annotations

import warnings
import os
from pathlib import Path

import numpy as np

from advassess import data as data_utils


def load_sklearn_estimator(cfg: dict):
    path = Path(cfg["target"]["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    if os.environ.get("ADVASSESS_ALLOW_UNSAFE_MODEL_LOADING") != "1":
        raise RuntimeError(
            "Refusing to load a joblib/pickle scikit-learn model by default because deserialization can execute code. "
            "For a trusted artifact only, set ADVASSESS_ALLOW_UNSAFE_MODEL_LOADING=1 and run in an isolated worker."
        )
    warnings.warn("UNSAFE MODEL LOADING ENABLED: joblib/pickle deserialization may execute arbitrary code.", stacklevel=2)
    import joblib  # installed together with scikit-learn; reads pickle files too

    model = joblib.load(path)
    if not hasattr(model, "predict") or not hasattr(model, "classes_"):
        raise TypeError("This file does not hold a fitted scikit-learn classifier (needs predict and classes_). "
                        f"Found: {type(model).__name__}")
    return model


def load_tabular_data(cfg: dict, classes: list):
    """(x, y) with y as indices into the model's classes."""
    d, seed = cfg["data"], cfg["run"]["seed"]
    if d["source"] == "csv":
        x, labels = data_utils.load_numeric_csv(d["path"], d["label_column"], d.get("feature_columns"),
                                                d["max_samples"], seed)
    elif d["source"] == "npz":
        x, labels = data_utils.load_npz(d["path"], None, d["max_samples"], seed)
        x = x.reshape(len(x), -1)
    else:
        raise ValueError("Tabular scikit-learn models need data.source: csv or npz")
    lookup = {c: i for i, c in enumerate(classes)}
    try:
        y = np.asarray([lookup[l.item() if hasattr(l, "item") else l] for l in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Label {exc.args[0]!r} in the data is not one of the model's classes {classes}") from None
    return x, y

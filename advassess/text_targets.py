"""Load a scikit-learn text classifier and a CSV test set.

Supported model files (saved with pickle or joblib):
  * a Pipeline that takes raw text, e.g. Pipeline([CountVectorizer, MultinomialNB])
  * a tuple, list or dict holding a fitted vectorizer (has .transform) and a
    fitted classifier (has .predict), e.g. (vectorizer, classifier)
"""
from __future__ import annotations

import csv
import warnings
import os
from pathlib import Path

import numpy as np


class SklearnTextClassifier:
    """Uniform black-box interface: raw text in, labels / probabilities out."""

    def __init__(self, obj) -> None:
        self._pipe = self._vec = self._clf = None
        if hasattr(obj, "predict") and hasattr(obj, "classes_") and _accepts_text(obj):
            self._pipe = obj
            self.classes_ = [_plain(c) for c in obj.classes_]
        else:
            parts = list(obj.values()) if isinstance(obj, dict) else list(obj) if isinstance(obj, (tuple, list)) else []
            vecs = [p for p in parts if hasattr(p, "transform") and not hasattr(p, "predict")]
            clfs = [p for p in parts if hasattr(p, "predict")]
            if len(vecs) != 1 or len(clfs) != 1:
                raise TypeError(
                    "Could not find the model in this file. Expected a scikit-learn Pipeline "
                    "that accepts raw text, or a (vectorizer, classifier) pair. "
                    f"Found: {type(obj).__name__}"
                )
            self._vec, self._clf = vecs[0], clfs[0]
            self.classes_ = [_plain(c) for c in self._clf.classes_]
        self.has_proba = hasattr(self._pipe or self._clf, "predict_proba")

    def predict(self, texts: list[str]) -> np.ndarray:
        if self._pipe is not None:
            return np.asarray(self._pipe.predict(texts))
        return np.asarray(self._clf.predict(self._vec.transform(texts)))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if self._pipe is not None:
            return np.asarray(self._pipe.predict_proba(texts))
        return np.asarray(self._clf.predict_proba(self._vec.transform(texts)))


def _plain(value):
    return value.item() if hasattr(value, "item") else value


def _accepts_text(model) -> bool:
    try:
        model.predict(["test"])
        return True
    except Exception:
        return False


def load_sklearn_model(cfg: dict) -> SklearnTextClassifier:
    path = Path(cfg["target"]["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    warnings.warn(
        "Loading a pickled model runs code stored in the file. "
        "Only load files from sources you trust.",
        stacklevel=2,
    )
    import joblib  # installed together with scikit-learn; reads pickle files too

    return SklearnTextClassifier(joblib.load(path))


def load_text_data(cfg: dict, classes: list) -> tuple[list[str], np.ndarray]:
    """Read the CSV and convert labels to the same type the model predicts."""
    d = cfg["data"]
    path = Path(d["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for col in (d["text_column"], d["label_column"]):
            if col not in (reader.fieldnames or []):
                raise KeyError(
                    f"Column '{col}' not found in {path.name}. Columns are: {reader.fieldnames}. "
                    "Set data.text_column / data.label_column in the config."
                )
        rows = [(r[d["text_column"]], r[d["label_column"]]) for r in reader if r[d["text_column"]]]

    sample = classes[0]
    if isinstance(sample, (int, np.integer)):
        convert = int
    elif isinstance(sample, (float, np.floating)):
        convert = float
    else:
        convert = str
    texts, labels = [], []
    for text, label in rows:
        try:
            value = convert(label.strip())
        except ValueError:
            raise ValueError(
                f"Label '{label}' in the CSV can't be matched to the model's classes {classes}"
            ) from None
        if value not in classes:
            raise ValueError(f"Label {value!r} in the CSV is not one of the model's classes {classes}")
        texts.append(text)
        labels.append(value)

    n = min(d["max_samples"], len(texts))
    idx = np.random.default_rng(cfg["run"]["seed"]).permutation(len(texts))[:n]
    return [texts[i] for i in idx], np.asarray([labels[i] for i in idx])

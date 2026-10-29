"""Load the model under test and the data to test it on.

Sources
-------
Models: 'torchscript'
        (a .pt file saved with torch.jit), 'pickle' (a full model saved with torch.save).
Data:   'npz' (a NumPy file with arrays x_test and y_test) or 'csv'.

The model must take a batch shaped (N, *input_shape) and return raw class scores
(logits) shaped (N, num_classes).
"""
from __future__ import annotations

import warnings
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn


def resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but no GPU is available to PyTorch")
    return choice


def load_model(cfg: dict) -> tuple[nn.Module, str]:
    """Return (model, device) ready for evaluation."""
    t = cfg["target"]
    device = resolve_device(t["device"])
    path = Path(t["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    if t["source"] == "state_dict":
        model = _build_model_class(t)
        try:
            raw = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read '{path.name}' as weights only ({type(exc).__name__}). If the file "
                "holds a whole saved model rather than weights, use target.source: pickle."
            ) from None
        model.load_state_dict(_unwrap_state_dict(raw))
    elif t["source"] == "torchscript":
        model = torch.jit.load(str(path), map_location=device)
    else:  # pickle
        if os.environ.get("ADVASSESS_ALLOW_UNSAFE_MODEL_LOADING") != "1":
            raise RuntimeError(
                "Refusing to deserialize a pickle-based PyTorch model by default. "
                "Prefer target.source: state_dict or torchscript. For a trusted artifact only, "
                "set ADVASSESS_ALLOW_UNSAFE_MODEL_LOADING=1 and run in an isolated worker."
            )
        warnings.warn(
            "UNSAFE MODEL LOADING ENABLED: deserializing this pickle may execute arbitrary code.",
            stacklevel=2,
        )
        try:
            model = torch.load(str(path), map_location=device, weights_only=False)
        except RuntimeError as exc:
            if "magic number" in str(exc):
                raise RuntimeError(
                    "This file is not a PyTorch model. If it is a scikit-learn model "
                    "(for example CountVectorizer + MultinomialNB), set target.source: sklearn."
                ) from None
            raise
    if not isinstance(model, nn.Module):
        raise TypeError("The loaded file is not a PyTorch model (nn.Module)")

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)  # ART computes input gradients itself; weights stay frozen
    return model, device


def load_data(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_test, y_test) with y_test as integer class labels."""
    from advassess import data as data_utils

    d = cfg["data"]
    seed = cfg["run"]["seed"]
    shape = cfg["target"].get("input_shape")
    if d["source"] == "csv":
        x, labels = data_utils.load_numeric_csv(d["path"], d["label_column"], d.get("feature_columns"),
                                                d["max_samples"], seed)
        if labels.dtype.kind not in "iu":
            raise ValueError("For PyTorch and Keras models the label column must hold class numbers 0, 1, 2 ...")
        if shape is not None and tuple(x.shape[1:]) != tuple(shape):
            raise ValueError(f"The csv has {x.shape[1]} features but target.input_shape is {tuple(shape)}")
        return x, labels.astype(np.int64)
    return data_utils.load_npz(d["path"], shape, d["max_samples"], seed)

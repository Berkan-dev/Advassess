"""Keras / TensorFlow models, attacked through ART."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


def _tf():
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is not installed. Install it with:  pip install tensorflow") from None
    return tf


def load_keras_model(cfg: dict):
    tf = _tf()
    path = Path(cfg["target"]["path"])
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    if path.is_dir():
        raise ValueError("This is a TensorFlow SavedModel folder, which Keras 3 cannot load as a model. "
                         "Re-save it from your training code with  model.save('model.keras')  and point "
                         "target.path at that file.")
    warnings.warn("Loading a saved model can run code stored in the file (for example Lambda layers). "
                  "Only load files from sources you trust.", stacklevel=2)
    model = tf.keras.models.load_model(str(path), compile=False)
    if isinstance(model.input_shape, list) or isinstance(model.output_shape, list):
        raise ValueError("Models with several inputs or outputs are not supported; use a single-input classifier.")
    return model


def model_shapes(model) -> tuple[tuple, int]:
    return tuple(int(d) for d in model.input_shape[1:]), int(model.output_shape[-1])


def detect_outputs(model, x_sample: np.ndarray) -> str:
    """'probabilities' if the model's outputs already sum to 1, otherwise 'logits'."""
    out = np.asarray(model(x_sample[:32], training=False))
    looks_prob = bool(out.min() >= -1e-6 and out.max() <= 1 + 1e-6 and np.allclose(out.sum(axis=1), 1.0, atol=1e-3))
    return "probabilities" if looks_prob else "logits"

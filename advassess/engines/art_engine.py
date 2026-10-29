"""Adversarial Robustness Toolbox (ART) engine for PyTorch classifiers.

Runs any attack listed in art_catalog.CATALOG. Attacks are looked up by short
name (fgsm, jsma, cw ...) or ART class name, built with the settings from the
config, and applied in chunks so large models and datasets fit in memory.
"""
from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import os
import time

import numpy as np

from advassess.engines import art_catalog as cat

os.environ.setdefault("TQDM_DISABLE", "1")  # ART draws progress bars that would flood the output


# Attacks that only work one input at a time (we also feed them one input per call).
BATCH_ONE = {"SimBA"}
SINGLE_SAMPLE = {"ShadowAttack"}
# Attacks that need the model to output probabilities instead of raw scores.
NEEDS_PROBABILITIES = {"VirtualAdversarialMethod", "SimBA"}
# Attacks that need raw scores (logits). For a model that outputs probabilities we feed them log(p),
# which is a valid set of logits (softmax(log p) = p).
NEEDS_LOGITS = {"AutoAttack"}
# Attacks that read y as the class to steer TOWARDS. Untargeted runs pass y=None
# so ART picks a random wrong class for each input.
Y_IS_TARGET = {"SaliencyMapMethod"}
# Needs colour images (3 channels).
NEEDS_RGB = {"CompositeAdversarialAttackPyTorch"}
# Settings added when the user gives none (these attacks do nothing without them).
DEFAULT_PARAMS = {
    "AutoConjugateGradient": {"loss_type": "cross_entropy"},
    "RescalingAutoConjugateGradient": {"loss_type": "cross_entropy"},
}


def known_attacks() -> set[str]:
    return cat.known_names()


_TORCH_WRAPPERS = None


def _torch_wrappers():
    """Small PyTorch wrapper modules, defined on first use so PyTorch stays optional."""
    global _TORCH_WRAPPERS
    if _TORCH_WRAPPERS is None:
        from torch import nn

        class Float32(nn.Module):
            """Some black-box attacks feed float64 arrays; PyTorch models expect float32."""

            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model(x.float())

        class Softmax(nn.Module):
            """Makes a model that outputs raw scores (logits) output probabilities."""

            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return nn.functional.softmax(self.model(x), dim=1)

        _TORCH_WRAPPERS = (Float32, Softmax)
    return _TORCH_WRAPPERS


def build_classifier(model, cfg: dict, device: str, probabilities: bool = False, logits: bool = False):
    """Wrap a loaded model (PyTorch, Keras or scikit-learn) as an ART classifier.

    probabilities=True gives a view whose outputs are probabilities; logits=True a view whose outputs are raw scores."""
    t = cfg["target"]
    clip = (float(t["clip_values"][0]), float(t["clip_values"][1]))
    if t["source"] == "keras":
        return _keras_classifier(model, t, clip, probabilities, logits)
    if t["source"] == "sklearn":
        from art.estimators.classification import SklearnClassifier

        return SklearnClassifier(model=model, clip_values=clip)

    from art.estimators.classification import PyTorchClassifier
    from torch import nn

    Float32, Softmax = _torch_wrappers()
    wrapped = Softmax(Float32(model)) if probabilities else Float32(model)
    return PyTorchClassifier(
        model=wrapped,
        loss=nn.CrossEntropyLoss(),
        input_shape=tuple(t["input_shape"]),
        nb_classes=t["num_classes"],
        clip_values=clip,
        device_type="gpu" if device == "cuda" else "cpu",
    )


def _keras_classifier(model, t: dict, clip: tuple, probabilities: bool, logits: bool = False):
    import tensorflow as tf
    from art.estimators.classification import TensorFlowV2Classifier

    outputs = t["_outputs"]  # resolved to logits or probabilities when the model was loaded
    net = model
    if probabilities and outputs == "logits":
        net = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        outputs = "probabilities"
    elif logits and outputs == "probabilities":
        net = tf.keras.Sequential([model, tf.keras.layers.Lambda(
            lambda p: tf.math.log(tf.clip_by_value(p, 1e-12, 1.0)))])
        outputs = "logits"
    return TensorFlowV2Classifier(
        model=net,
        nb_classes=t["num_classes"],
        input_shape=tuple(t["input_shape"]),
        loss_object=tf.keras.losses.CategoricalCrossentropy(from_logits=(outputs == "logits")),
        clip_values=clip,
        channels_first=False,
    )


def _pick_view(class_name: str, classifier, prob_classifier, logit_classifier):
    if class_name in NEEDS_PROBABILITIES:
        return prob_classifier
    if class_name in NEEDS_LOGITS and logit_classifier is not None:
        return logit_classifier
    return classifier


def check_applicable(name: str, classifier, prob_classifier, params: dict, batch_size: int,
                     logit_classifier=None) -> str | None:
    """None if the attack can be built for this model, otherwise the reason it can't (used for default suites)."""
    class_name = cat.resolve_name(name)
    model = _pick_view(class_name, classifier, prob_classifier, logit_classifier)
    try:
        build_attack(name, model, params, batch_size)
    except RuntimeError as exc:
        if "can't run on this model type" in str(exc):
            return "not applicable to this model type"
        return str(exc)[:120]
    except ModuleNotFoundError as exc:
        return _missing_package_message(exc, class_name)
    except Exception:
        return None  # let the real run report anything else
    return None


def _load_class(class_name: str):
    module = importlib.import_module("art.attacks.evasion")
    try:
        return getattr(module, class_name)
    except AttributeError:
        raise RuntimeError(
            f"Your installed version of ART has no attack called {class_name}. "
            "Update it with: pip install -U adversarial-robustness-toolbox"
        ) from None


def describe(name: str) -> dict:
    """Class, category and settings (with defaults) for one attack."""
    class_name = cat.resolve_name(name)
    if class_name is None:
        raise KeyError(name)
    cls = _load_class(class_name)
    params = []
    for pname, p in inspect.signature(cls.__init__).parameters.items():
        if pname in ("self", "estimator", "classifier") or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        params.append((pname, None if p.default is inspect.Parameter.empty else p.default,
                       p.default is inspect.Parameter.empty))
    doc = (inspect.getdoc(cls) or "").strip().splitlines()
    return {"class": class_name, "category": cat.CATALOG[class_name][0],
            "summary": cat.CATALOG[class_name][1], "params": params,
            "doc": doc[0] if doc else "", "tested": class_name in cat.VERIFIED}


def _clean_params(params: dict) -> dict:
    """Translate friendly config values (e.g. norm: inf) into what ART expects."""
    out = dict(params)
    if "norm" in out:
        n = out["norm"]
        out["norm"] = np.inf if str(n).lower() in {"inf", "infinity", "linf"} else int(n)
    return out


def build_attack(name: str, classifier, params: dict, batch_size: int):
    class_name = cat.resolve_name(name)
    cls = _load_class(class_name)
    accepted = inspect.signature(cls.__init__).parameters
    kwargs = {**DEFAULT_PARAMS.get(class_name, {}), **_clean_params(params)}
    if "targeted" in accepted:
        kwargs.setdefault("targeted", False)  # a few ART attacks default to targeted; this tool defaults to untargeted
    if "batch_size" in accepted:
        kwargs.setdefault("batch_size", batch_size)
        if class_name in BATCH_ONE or (class_name == "ZooAttack" and len(classifier.input_shape) == 1):
            kwargs["batch_size"] = 1  # ZOO handles feature vectors one input at a time
    if "verbose" in accepted:
        kwargs.setdefault("verbose", False)
    unknown = [k for k in kwargs if k not in accepted]
    if unknown:
        raise ValueError(f"{class_name} does not accept parameter(s): {unknown}. "
                         f"See them with: python run.py --describe-attack {name}")
    # older ART attacks call their model argument "classifier", newer ones "estimator"
    model_arg = next(p for p in accepted if p != "self")
    try:
        return cls(**{model_arg: classifier}, **kwargs)
    except Exception as exc:
        if type(exc).__name__ == "EstimatorError":
            raise RuntimeError(f"{class_name} can't run on this model type: {exc}") from None
        raise


def _missing_package_message(exc: ModuleNotFoundError, class_name: str) -> str:
    pkg = (exc.name or "").split(".")[0]
    return f"{class_name} needs an extra package. Install it with: pip install {pkg}"


def run_attack(
    name: str,
    params: dict,
    classifier,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    n_examples: int = 8,
    samples: int | None = None,
    prob_classifier=None,
    logit_classifier=None,
) -> dict:
    """Attack x in chunks and measure the effect.

    Untargeted by default. For a targeted attack set `targeted: true` and
    `target_class: N` in the params; success then means the model outputs class N.
    """
    params = dict(params)
    targeted = bool(params.get("targeted", False))
    target_class = params.pop("target_class", None)
    if targeted and target_class is None:
        raise ValueError("A targeted attack needs target_class: <class number> in its params")
    if not targeted and target_class is not None:
        raise ValueError("target_class only applies with targeted: true")

    if samples:
        x, y = x[:samples], y[:samples]
    class_name = cat.resolve_name(name)
    accepts_targeted = "targeted" in inspect.signature(_load_class(class_name).__init__).parameters
    if targeted and not accepts_targeted and class_name not in Y_IS_TARGET:
        raise ValueError(f"{class_name} has no targeted mode")
    build_params = {k: v for k, v in params.items() if not (k == "targeted" and class_name in Y_IS_TARGET)}
    if class_name in NEEDS_RGB and tuple(classifier.input_shape)[0] != 3:
        raise RuntimeError(f"{class_name} needs colour images with 3 channels (input_shape [3, H, W])")
    if class_name in NEEDS_PROBABILITIES and prob_classifier is None:
        raise RuntimeError(f"{class_name} needs a probability-output model wrapper (prob_classifier)")
    attack_model = _pick_view(class_name, classifier, prob_classifier, logit_classifier)
    try:
        attack = build_attack(name, attack_model, build_params, batch_size)
    except ModuleNotFoundError as exc:
        raise RuntimeError(_missing_package_message(exc, class_name)) from None
    no_verbose_flag = "verbose" not in inspect.signature(_load_class(class_name).__init__).parameters
    n_classes = classifier.nb_classes
    eye = np.eye(n_classes, dtype=np.float32)
    chunk = batch_size * 4  # only one chunk of adversarial data is held in memory at a time

    n = len(x)
    clean_correct = adv_correct = fooled = eligible = 0
    l2_sum = linf_sum = 0.0
    examples: list[tuple] = []
    started = time.perf_counter()

    for start in range(0, n, chunk):
        xb, yb = x[start:start + chunk], y[start:start + chunk]
        clean_pred = classifier.predict(xb, batch_size=batch_size).argmax(axis=1)
        goal = np.full_like(yb, int(target_class)) if targeted else yb
        try:
            y_arg = None if (class_name in Y_IS_TARGET and not targeted) else eye[goal]
            quiet = contextlib.redirect_stderr(io.StringIO()) if no_verbose_flag else contextlib.nullcontext()
            with quiet:
              if class_name in SINGLE_SAMPLE:
                x_adv = np.concatenate([
                    attack.generate(x=xb[i:i + 1], y=None if y_arg is None else y_arg[i:i + 1])
                    for i in range(len(xb))])
              else:
                x_adv = attack.generate(x=xb, y=y_arg)
        except ModuleNotFoundError as exc:
            raise RuntimeError(_missing_package_message(exc, class_name)) from None
        x_adv = np.where(np.isfinite(x_adv), x_adv, xb).astype(np.float32)  # a failed sample stays unchanged
        adv_pred = classifier.predict(x_adv, batch_size=batch_size).argmax(axis=1)

        was_ok = clean_pred == yb
        can_attack = was_ok & (yb != goal) if targeted else was_ok
        hit = can_attack & ((adv_pred == goal) if targeted else (adv_pred != yb))
        clean_correct += int(was_ok.sum())
        adv_correct += int((adv_pred == yb).sum())
        eligible += int(can_attack.sum())
        fooled += int(hit.sum())

        diff = (x_adv - xb).reshape(len(xb), -1)
        l2_sum += float(np.linalg.norm(diff, ord=2, axis=1).sum())
        linf_sum += float(np.abs(diff).max(axis=1).sum())

        for i in np.flatnonzero(hit):
            if len(examples) < n_examples:
                before, after = xb[i], x_adv[i]
                if getattr(classifier, "channels_first", True) is False and before.ndim == 3:
                    before, after = np.transpose(before, (2, 0, 1)), np.transpose(after, (2, 0, 1))
                examples.append((before, after, int(yb[i]), int(adv_pred[i])))

    return {
        "attack": name.upper(),
        "art_class": class_name,
        "category": cat.CATALOG[class_name][0],
        "params": {**params, **({"target_class": int(target_class)} if targeted else {})},
        "n_samples": n,
        "clean_accuracy": clean_correct / n,
        "adversarial_accuracy": adv_correct / n,
        "attack_success_rate": (fooled / eligible) if eligible else None,
        "mean_l2": l2_sum / n,
        "mean_linf": linf_sum / n,
        "seconds": round(time.perf_counter() - started, 2),
        "examples": examples,
        "error": None,
    }

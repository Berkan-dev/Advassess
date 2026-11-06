"""ART attacks that do not fit "attack one input of a classifier at a time".

Three kinds, selected by art_catalog.scope_of / task_of:

dataset   Learn ONE perturbation or patch on a fit split, then score it on held-out inputs it never saw
          (UniversalPerturbation, TargetedUniversalPerturbation, AdversarialPatch*). The result is the success
          rate of that single shared perturbation on the held-out split, which is what these attacks claim.
setup     Attacks that need something the config would not normally say. The tool derives it from the data and
          the model, and every derived value can be overridden under params (LowProFool, FeatureAdversaries*,
          GRAPHITE*, LaserBeamAttack).
adapter   Attacks on other model types (object detection, tracking, video, speech, malware, Gaussian process).
          Their models, inputs and success measures cannot be described by a plain config, so a small Python
          adapter you write supplies them (see ADAPTER_HELP).

Every function returns the same result dict as art_engine.run_attack, plus a few extra keys.
"""
from __future__ import annotations

import copy
import importlib.util
import inspect
import math
import time
from pathlib import Path

import numpy as np

from advassess.engines import art_catalog as cat

MIN_SAMPLES_DATASET = 20   # a shared perturbation needs a fit split and a held-out split worth scoring

ADAPTER_HELP = """\
An adapter is a Python class in a file you write and trust (it runs with your permissions, like target.model_class):

    class MyAdapter:
        def build_estimator(self):
            "return the ART estimator for your model (PyTorchYolo, PyTorchDeepSpeech, ...)"
        def inputs(self):
            "return the keyword arguments for attack.generate(), e.g. {'x': images, 'y': labels}"
        def score(self, estimator, inputs, generated, attack):
            "return {'clean_score': .., 'adv_score': .., 'success_rate': .., 'n': ..}, all between 0 and 1"
        # optional: def attack_kwargs(self, attack_name) -> dict   (extra constructor settings)

Point the config at it:  target: {source: adapter, adapter: "my_adapter.py:MyAdapter"}.
'score' is the success measure for your task (for detection, the share of objects no longer found; for speech, the
share of utterances transcribed as the attacker's phrase, ...). The tool cannot guess it."""


# --------------------------------------------------------------------------- shared helpers
def _eye(n_classes: int) -> np.ndarray:
    return np.eye(n_classes, dtype=np.float32)


def _clip_to(classifier, xb: np.ndarray) -> np.ndarray:
    cv = getattr(classifier, "clip_values", None)
    return np.clip(xb, cv[0], cv[1]) if cv is not None else xb


def _predict(classifier, xb: np.ndarray, batch_size: int) -> np.ndarray:
    return classifier.predict(xb, batch_size=batch_size).argmax(axis=1)


def _pick_other_class(y: np.ndarray, n_classes: int, rng: np.random.Generator) -> np.ndarray:
    """For each label, a different class that exists in the data (so a guide or target example can be found)."""
    present = np.unique(y)
    if len(present) < 2:
        raise ValueError("This attack needs test data with at least two different classes")
    out = np.empty_like(y)
    for i, c in enumerate(y):
        out[i] = rng.choice(present[present != c])
    return out


def _guides_for(x: np.ndarray, y: np.ndarray, goal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One input of class goal[i] for every i, drawn from the test data (a 'guide' or 'target' example)."""
    by_class = {int(c): np.flatnonzero(y == c) for c in np.unique(y)}
    return x[np.array([rng.choice(by_class[int(g)]) for g in goal])]


def _summarise(name, class_name, params, classifier, xe, ye, x_adv, goal, targeted, batch_size, n_examples,
               started, **extra) -> dict:
    """Standard result row from held-out inputs xe (labels ye), their adversarial versions and the attack goal."""
    x_adv = np.where(np.isfinite(x_adv), x_adv, xe).astype(np.float32)
    clean_pred = _predict(classifier, xe, batch_size)
    adv_pred = _predict(classifier, x_adv, batch_size)
    was_ok = clean_pred == ye
    can_attack = was_ok & (ye != goal) if targeted else was_ok
    hit = can_attack & ((adv_pred == goal) if targeted else (adv_pred != ye))
    diff = (x_adv - xe).reshape(len(xe), -1)
    examples = []
    for i in np.flatnonzero(hit)[:n_examples]:
        before, after = xe[i], x_adv[i]
        if getattr(classifier, "channels_first", True) is False and before.ndim == 3:
            before, after = np.transpose(before, (2, 0, 1)), np.transpose(after, (2, 0, 1))
        examples.append((before, after, int(ye[i]), int(adv_pred[i])))
    eligible = int(can_attack.sum())
    return {
        "attack": name.upper(), "art_class": class_name, "category": cat.CATALOG[class_name][0],
        "params": params, "n_samples": len(xe),
        "clean_accuracy": float(was_ok.mean()), "adversarial_accuracy": float((adv_pred == ye).mean()),
        "attack_success_rate": (int(hit.sum()) / eligible) if eligible else None,
        "mean_l2": float(np.linalg.norm(diff, axis=1).mean()), "mean_linf": float(np.abs(diff).max(axis=1).mean()),
        "seconds": round(time.perf_counter() - started, 2), "examples": examples, "error": None, **extra,
    }


def _success_rate(classifier, xb, yb, x_adv, goal, targeted, batch_size) -> float | None:
    clean_pred, adv_pred = _predict(classifier, xb, batch_size), _predict(classifier, x_adv, batch_size)
    was_ok = clean_pred == yb
    can = was_ok & (yb != goal) if targeted else was_ok
    hit = can & ((adv_pred == goal) if targeted else (adv_pred != yb))
    return (int(hit.sum()) / int(can.sum())) if can.any() else None


def _targeting(params: dict, n_classes: int) -> tuple[bool, int | None]:
    targeted = bool(params.get("targeted", False))
    target_class = params.pop("target_class", None)
    if targeted and target_class is None:
        raise ValueError("A targeted attack needs target_class: <class number> in its params")
    if not targeted and target_class is not None:
        raise ValueError("target_class only applies with targeted: true")
    if targeted and not 0 <= int(target_class) < n_classes:
        raise ValueError(f"target_class must be between 0 and {n_classes - 1}")
    return targeted, (int(target_class) if targeted else None)


# --------------------------------------------------------------------------- dataset scope
def run_dataset_attack(name, params, classifier, x, y, batch_size, n_examples=8, samples=None,
                       build_attack=None, **_unused) -> dict:
    """One shared perturbation or patch: learn on the first part of the data, score on the rest.

    Extra settings under params:  fit_fraction (default 0.5, share of the inputs used to learn it),
    apply_scale (patch attacks, default 0.4, patch size relative to the input) and the attack's own settings.
    """
    started = time.perf_counter()
    params = dict(params)
    class_name = cat.resolve_name(name)
    fit_fraction = float(params.pop("fit_fraction", 0.5))
    apply_scale = float(params.pop("apply_scale", 0.4))
    if not 0.1 <= fit_fraction <= 0.9:
        raise ValueError("fit_fraction must be between 0.1 and 0.9")
    if samples:
        x, y = x[:samples], y[:samples]
    if len(x) < MIN_SAMPLES_DATASET:
        raise ValueError(f"{class_name} learns one shared perturbation on part of the data and scores it on the rest, "
                         f"so it needs at least {MIN_SAMPLES_DATASET} test samples (got {len(x)}). "
                         "Raise data.max_samples or the attack's 'samples'.")
    n_classes = classifier.nb_classes
    targeted, target_class = _targeting(params, n_classes)
    is_patch = class_name.startswith("AdversarialPatch")
    if class_name == "TargetedUniversalPerturbation" and not targeted:
        raise ValueError("TargetedUniversalPerturbation is a targeted attack: set targeted: true and target_class: N")
    if is_patch and len(classifier.input_shape) != 3:
        raise ValueError(f"{class_name} places a patch on an image, but this model takes inputs shaped "
                         f"{tuple(classifier.input_shape)}, not (channels, height, width)")
    if class_name == "AdversarialPatchNumpy" and targeted:
        params.setdefault("target", target_class)
    build_params = dict(params)
    if class_name == "TargetedUniversalPerturbation":
        build_params.pop("targeted", None)  # always targeted; ART's class has no such setting
        # the inner attack must also steer towards the target class, or it pushes the other way
        build_params["attacker_params"] = {"targeted": True, **(build_params.get("attacker_params") or {})}
    attack = build_attack(name, classifier, build_params, batch_size)

    n_fit = max(1, int(round(len(x) * fit_fraction)))
    xf, yf, xe, ye = x[:n_fit], y[:n_fit], x[n_fit:], y[n_fit:]
    eye = _eye(n_classes)
    goal_f = np.full_like(yf, target_class) if targeted else yf
    goal_e = np.full_like(ye, target_class) if targeted else ye

    if is_patch:
        attack.generate(x=xf, y=eye[goal_f])

        def apply(xb):
            return np.concatenate([attack.apply_patch(xb[i:i + batch_size * 4], scale=apply_scale)
                                   for i in range(0, len(xb), batch_size * 4)])
        what = "patch"
    else:
        x_adv_f = attack.generate(x=xf, y=eye[goal_f] if targeted else None)
        noise = getattr(attack, "noise", None)
        if noise is None:
            noise = getattr(attack, "_noise", None)
        if noise is None:  # fall back to the typical difference (clipping can make single inputs differ)
            noise = np.median(x_adv_f - xf, axis=0, keepdims=True)
        noise = np.asarray(noise, dtype=np.float32)

        def apply(xb):
            return _clip_to(classifier, xb + noise)
        what = "perturbation"

    fit_rate = _success_rate(classifier, xf, yf, apply(xf), goal_f, targeted, batch_size)
    x_adv_e = apply(xe)
    return _summarise(
        name, class_name, {**params, "fit_fraction": fit_fraction, **({"apply_scale": apply_scale} if is_patch else {}),
                           **({"target_class": target_class} if targeted else {})},
        classifier, xe, ye, x_adv_e, goal_e, targeted, batch_size, n_examples, started,
        scope="dataset", n_fit=len(xf), fit_success_rate=fit_rate,
        note=f"One shared {what} was learned on {len(xf)} inputs and scored on {len(xe)} held-out inputs it never "
             f"saw. Its success rate on the fit inputs was "
             f"{'n/a' if fit_rate is None else f'{100 * fit_rate:.1f}%'}; a large gap means it does not generalise.")


# --------------------------------------------------------------------------- setup scope
def _feature_adversaries_class(classifier) -> str:
    kind = type(classifier).__name__
    if kind == "PyTorchClassifier":
        return "FeatureAdversariesPyTorch"
    if kind == "TensorFlowV2Classifier":
        return "FeatureAdversariesTensorFlowV2"
    return "FeatureAdversariesNumpy"


def run_setup_attack(name, params, classifier, x, y, batch_size, n_examples=8, samples=None, build_attack=None,
                     prob_classifier=None, seed=0, **_unused) -> dict:
    started = time.perf_counter()
    params = dict(params)
    class_name = cat.resolve_name(name)
    if samples:
        x, y = x[:samples], y[:samples]
    n_classes = classifier.nb_classes
    rng = np.random.default_rng(seed)
    lo, hi = (classifier.clip_values if getattr(classifier, "clip_values", None) is not None
              else (float(x.min()), float(x.max())))
    lo, hi = float(np.min(lo)), float(np.max(hi))
    eye = _eye(n_classes)
    targeted_cfg = bool(params.pop("targeted", True))   # all of these steer towards a derived target
    explicit_target = params.pop("target_class", None)

    def goals():
        if explicit_target is not None:
            return np.full_like(y, int(explicit_target))
        return _pick_other_class(y, n_classes, rng)

    if class_name.startswith("FeatureAdversaries"):
        class_name = _feature_adversaries_class(classifier)
        goal = goals()
        guides = _guides_for(x, y, goal, rng)
        params.setdefault("delta", float(f"{0.1 * (hi - lo):.4g}"))
        if class_name == "FeatureAdversariesNumpy":
            names = getattr(classifier, "layer_names", None) or []
            if "layer" not in params and len(names) >= 2:
                params["layer"] = len(names) - 2  # the representation just before the output layer
        attack = build_attack(class_name, classifier, params, batch_size)
        x_adv = attack.generate(x=x, y=guides)
        used = {**params, "guides": "inputs of another class from the test data"}

    elif class_name == "LowProFool":
        if len(classifier.input_shape) != 1:
            raise ValueError("LowProFool is for tabular models (one-dimensional feature vectors)")
        model = prob_classifier or classifier           # it needs class probabilities
        mins, maxs = x.min(axis=0), x.max(axis=0)       # per-feature valid range, taken from the data
        model = copy.copy(model)
        model._clip_values = (mins.astype(np.float32), maxs.astype(np.float32))
        importance = params.pop("importance_array", None)
        params.setdefault("importance", "pearson")
        params.setdefault("n_steps", 100)
        attack = build_attack(name, model, params, batch_size)
        attack.fit_importances(x=x, y=y.astype(np.float64), importance_array=None if importance is None
                               else np.asarray(importance, dtype=np.float64))
        probs = model.predict(x, batch_size=batch_size)
        order = np.argsort(-probs, axis=1)
        goal = (np.full_like(y, int(explicit_target)) if explicit_target is not None
                else np.where(order[:, 0] == y, order[:, 1], order[:, 0]))  # the runner-up class
        x_adv = attack.generate(x=x, y=eye[goal])
        used = {**params, "importance": "given" if importance is not None else params["importance"],
                "feature_ranges": "from the test data"}
        class_name = "LowProFool"

    elif class_name in ("GRAPHITEBlackbox", "GRAPHITEWhiteboxPyTorch"):
        if len(classifier.input_shape) != 3:
            raise ValueError(f"{class_name} works on images, but this model takes inputs shaped "
                             f"{tuple(classifier.input_shape)}")
        shape = tuple(classifier.input_shape)
        h, w = (shape[1], shape[2]) if getattr(classifier, "channels_first", True) is not False else (shape[0], shape[1])
        params.setdefault("net_size", (w, h))
        if class_name == "GRAPHITEBlackbox":
            params.setdefault("noise_size", (w, h))
        goal = goals()
        tars = _guides_for(x, y, goal, rng)
        attack = build_attack(class_name, classifier, params, batch_size)
        x_adv = attack.generate(x=x, y=eye[goal], x_tar=tars)
        used = {**params, "x_tar": "inputs of the target class from the test data"}

    elif class_name == "LaserBeamAttack":
        if getattr(classifier, "channels_first", True) is not False:
            raise ValueError("LaserBeamAttack needs a channels-last model (Keras / TensorFlow): it feeds NHWC images "
                             "straight to the model")
        h, w = classifier.input_shape[0], classifier.input_shape[1]
        params.setdefault("iterations", 5)
        params.setdefault("max_laser_beam", (780.0, math.pi, float(max(h, w)), float(max(2.0, w / 8))))
        params["max_laser_beam"] = tuple(params["max_laser_beam"])
        if "min_laser_beam" in params:
            params["min_laser_beam"] = tuple(params["min_laser_beam"])
        attack = build_attack(class_name, classifier, params, batch_size)
        x_adv = attack.generate(x=x, y=y)
        goal, targeted_cfg = y, False                   # untargeted: any change of the model's answer
        used = {**params}

    else:
        raise RuntimeError(f"{class_name} has no setup driver")

    return _summarise(name, class_name, _jsonable(used), classifier, x, y, x_adv, goal, targeted_cfg, batch_size,
                      n_examples, started, scope="setup",
                      note="Settings the config did not give were derived from the data and model; they are listed in "
                           "params and can be overridden there.")


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (tuple, list)):
            out[k] = [float(i) if isinstance(i, (int, float, np.floating)) else str(i) for i in v]
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = str(v)[:80]
    return out


# --------------------------------------------------------------------------- adapter (other model types)
def load_adapter(spec: str):
    """Instantiate 'file.py:ClassName'. Runs the user's own file, the same trust level as target.model_class."""
    path, _, cls = spec.partition(":")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Adapter file not found: {p}")
    module_spec = importlib.util.spec_from_file_location("advassess_user_adapter", p)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if not hasattr(module, cls):
        raise AttributeError(f"{p.name} has no class {cls!r}")
    adapter = getattr(module, cls)()
    missing = [m for m in ("build_estimator", "inputs", "score") if not callable(getattr(adapter, m, None))]
    if missing:
        raise TypeError(f"The adapter must define {', '.join(missing)}.\n\n{ADAPTER_HELP}")
    return adapter


def run_adapter_attack(name, params, adapter, estimator, inputs, batch_size, build_attack, **_unused) -> dict:
    started = time.perf_counter()
    class_name = cat.resolve_name(name)
    params = dict(params)
    extra = getattr(adapter, "attack_kwargs", None)
    if callable(extra):
        for k, v in (extra(class_name) or {}).items():
            params.setdefault(k, v)
    attack = build_attack(name, estimator, params, batch_size)
    accepted = inspect.signature(attack.generate).parameters
    generated = attack.generate(**{k: v for k, v in inputs.items() if k in accepted or "kwargs" in accepted})
    metrics = adapter.score(estimator, inputs, generated, attack)
    for key in ("clean_score", "adv_score", "success_rate"):
        if key not in metrics or not 0.0 <= float(metrics[key]) <= 1.0:
            raise ValueError(f"adapter.score() must return '{key}' as a number between 0 and 1 (got {metrics.get(key)!r})")
    x = inputs.get("x")
    measured = isinstance(generated, np.ndarray) and isinstance(x, np.ndarray) and generated.shape == x.shape
    if measured:
        diff = (generated - x).reshape(len(x), -1)
        l2, linf = float(np.linalg.norm(diff, axis=1).mean()), float(np.abs(diff).max(axis=1).mean())
    else:
        l2 = linf = 0.0
    return {
        "attack": name.upper(), "art_class": class_name, "category": cat.CATALOG[class_name][0],
        "params": _jsonable(params), "n_samples": int(metrics.get("n", len(x) if x is not None else 0)),
        "clean_accuracy": float(metrics["clean_score"]), "adversarial_accuracy": float(metrics["adv_score"]),
        "attack_success_rate": float(metrics["success_rate"]),
        "mean_l2": l2, "mean_linf": linf, "perturbation_measured": measured,
        "seconds": round(time.perf_counter() - started, 2), "examples": [], "error": None,
        "scope": "adapter", "task": cat.task_of(class_name),
        "note": "Scores come from the adapter's own score(); the tool cannot check them. "
                + ("" if measured else "This attack returns something other than perturbed inputs (a patch, for "
                   "example), so L2 and Linf are not reported."),
    }

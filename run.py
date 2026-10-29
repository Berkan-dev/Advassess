"""Command-line entry point:  python run.py --config config.yaml"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import traceback

import numpy as np

from advassess.config import ConfigError, load_config, target_kind
from advassess.engines import text_engine
from advassess.report import format_table, limits_for, write_outputs


def _image_engine_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("art") is not None


def _registry() -> dict[str, set[str]]:
    from advassess.engines import art_catalog
    # the image engine needs PyTorch and ART; text-only users don't
    return {"text": text_engine.known_attacks(),
            "image": art_catalog.known_names() if _image_engine_available() else set()}


def _print_attack_list() -> None:
    from advassess.engines import art_catalog as cat
    names = {}
    for alias, cls in cat.ALIASES.items():
        names.setdefault(cls, []).append(alias)
    print("PyTorch classifiers (ART engine)")
    print(f"  {'name(s) to use in the config':34s}{'type':11s}{'tested':8s}what it does")
    for cls, (category, summary) in cat.CATALOG.items():
        short = ", ".join(sorted(names.get(cls, []))) or cls.lower()
        print(f"  {short:34s}{category:11s}{'yes' if cls in cat.VERIFIED else 'no':8s}{summary}")
    print("\nLanguage models (Garak): python run.py --list-probes")
    print("\nscikit-learn text models")
    print("  " + ", ".join(sorted(text_engine.known_attacks())))
    print("\nSee an attack's settings:  python run.py --describe-attack NAME")
    print("Attacks that need other model types (speech, video, object detection ...) are not supported.")


def _print_attack_description(name: str) -> int:
    from advassess.engines import art_engine
    try:
        d = art_engine.describe(name)
    except KeyError:
        print(f"Unknown attack '{name}'. See: python run.py --list-attacks")
        return 2
    print(f"{d['class']}  ({d['category']}, {'tested' if d['tested'] else 'not yet tested'})")
    print(d["summary"])
    print("\nSettings you can put under params (default in brackets, * = required):")
    for pname, default, required in d["params"]:
        print(f"  {pname}  [{'*' if required else default}]")
    print("\nExtra option for any attack: put 'samples: N' next to 'name' to limit how many test "
          "inputs it runs on (useful for slow attacks).")
    print("Targeted attacks: set targeted: true and target_class: N in params.")
    return 0


DEFAULT_SUITE_TIMEOUT = 180  # seconds per attack when the default suite runs


def _with_timeout(seconds, fn, *args):
    """Run fn, raising TimeoutError if it takes longer than `seconds` (not available on Windows)."""
    if not seconds or not hasattr(signal, "SIGALRM"):
        return fn(*args)

    def handler(signum, frame):
        raise TimeoutError(f"took longer than {seconds:g} s (raise run.attack_timeout_seconds to allow more)")

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _fill_default_attacks(cfg: dict, suite: list[dict]) -> None:
    cfg["attacks"] = [{"params": {}, **a} for a in suite]
    print(f"No attacks listed in the config, so the default suite runs ({len(cfg['attacks'])} attacks). "
          "To choose your own, list them under 'attacks'.")


def _run_art(cfg: dict):
    """PyTorch, Keras and scikit-learn numeric models, attacked through ART."""
    import art
    from advassess import data as data_utils
    from advassess import defaults
    from advassess.engines import art_engine

    t, src = cfg["target"], cfg["target"]["source"]
    versions = {"art": art.__version__, "numpy": np.__version__}
    device = "cpu"
    print("Loading model and data ...")

    if src == "keras":
        import tensorflow as tf
        from advassess import keras_targets as kt
        model = kt.load_keras_model(cfg)
        shape, n_classes = kt.model_shapes(model)
        x, y = data_utils.load_numeric(cfg)
        if tuple(x.shape[1:]) != shape:
            raise ValueError(f"The data has shape {tuple(x.shape[1:])} per sample but the Keras model expects "
                             f"{shape} (Keras uses channels-last, e.g. 28x28x1).")
        t["_outputs"] = kt.detect_outputs(model, x) if t["outputs"] == "auto" else t["outputs"]
        print(f"Keras model outputs: {t['_outputs']}" + (" (auto-detected)" if t["outputs"] == "auto" else ""))
        versions["tensorflow"] = tf.__version__
    elif src == "sklearn":  # numeric (tabular) scikit-learn model
        import sklearn
        from advassess import tabular_targets as tt
        model = tt.load_sklearn_estimator(cfg)
        x, y = tt.load_tabular_data(cfg, [c.item() if hasattr(c, "item") else c for c in model.classes_])
        shape, n_classes = (x.shape[1],), len(model.classes_)
        versions["scikit-learn"] = sklearn.__version__
    else:  # PyTorch
        import torch
        from advassess.targets import load_data, load_model
        torch.manual_seed(cfg["run"]["seed"])
        model, device = load_model(cfg)
        x, y = load_data(cfg)
        shape = tuple(x.shape[1:])
        n_classes = t["num_classes"]
        if n_classes is None:
            with torch.no_grad():
                n_classes = int(model(torch.from_numpy(x[:2]).to(device)).shape[-1])
        versions["torch"] = torch.__version__

    if t["input_shape"] is None:
        t["input_shape"] = list(shape)
    if t["num_classes"] is None:
        t["num_classes"] = n_classes
    if t["clip_values"] is None:
        lo, hi = data_utils.infer_clip_values(x)
        t["clip_values"] = [lo, hi]
        print(f"Valid input range not set in the config; using [{lo:g}, {hi:g}] inferred from the data "
              "(set target.clip_values to override).")
    lo, hi = t["clip_values"]
    classifier = art_engine.build_classifier(model, cfg, device)
    prob_classifier = art_engine.build_classifier(model, cfg, device, probabilities=True)
    logit_classifier = (art_engine.build_classifier(model, cfg, device, logits=True)
                        if src == "keras" and t.get("_outputs") == "probabilities" else None)
    bs = cfg["run"]["batch_size"]
    print(f"Device: {device} | samples: {len(x)} | inputs per sample: {tuple(x.shape[1:])} | classes: {n_classes} "
          f"| batch size: {bs}\n")

    if cfg.get("_default_attacks"):
        suite = defaults.art_suite(lo, hi, int(np.prod(shape)))
        if art_engine.check_applicable("fgsm", classifier, prob_classifier, {"eps": 0.1}, bs):
            print("This model exposes no gradients, so the query-only attacks are added to the default suite.")
            suite += defaults.blackbox_extra(lo, hi)
        keep, skipped = [], []
        for a in suite:
            why = art_engine.check_applicable(a["name"], classifier, prob_classifier, a["params"], bs,
                                              logit_classifier)
            (skipped if why else keep).append({**a, "reason": why} if why else a)
        if not keep:
            raise RuntimeError("None of the default attacks apply to this model type. See python run.py "
                               "--list-attacks and choose attacks under 'attacks'.")
        _fill_default_attacks(cfg, keep)
        cfg["_skipped"] = [{"attack": a["name"].upper(), "params": a["params"], "reason": a["reason"]} for a in skipped]
        if skipped:
            print("Skipped (cannot run on this model type): " + ", ".join(sorted({s["attack"] for s in cfg["_skipped"]})))

    def run_one(atk, idx=0):
        return art_engine.run_attack(
            atk["name"], atk["params"], classifier, x, y,
            batch_size=bs, n_examples=cfg["output"]["save_examples"],
            samples=atk.get("samples"), prob_classifier=prob_classifier,
            logit_classifier=logit_classifier)

    return run_one, versions, device


def _run_text(cfg: dict):
    import sklearn
    from advassess.text_targets import load_sklearn_model, load_text_data

    from advassess import defaults
    print("Loading model and data ...")
    if cfg.get("_default_attacks"):
        _fill_default_attacks(cfg, defaults.TEXT_SUITE)
    clf = load_sklearn_model(cfg)
    texts, labels = load_text_data(cfg, clf.classes_)
    print(f"Classes: {clf.classes_} | samples: {len(texts)} | "
          f"confidence scores available: {'yes' if clf.has_proba else 'no (weaker attacks)'}\n")

    def run_one(atk, idx=0):
        n = atk.get("samples") or len(texts)
        return text_engine.run_attack(atk["name"], atk["params"], clf, texts[:n], labels[:n],
                                      n_examples=cfg["output"]["save_examples"])

    return run_one, {"scikit-learn": sklearn.__version__, "numpy": np.__version__}, "cpu"


def _run_llm(cfg: dict):
    from datetime import datetime, timezone
    from pathlib import Path
    from advassess import defaults
    from advassess.engines import garak_engine as g
    from advassess.engines import pyrit_engine as pe

    llm, run = cfg["target"]["llm"], cfg["run"]
    if cfg.get("_default_attacks"):
        _fill_default_attacks(cfg, defaults.LLM_SUITE)
    garak_entries = [a for a in cfg["attacks"] if a.get("engine") != "pyrit"]
    pyrit_entries = [a for a in cfg["attacks"] if a.get("engine") == "pyrit"]
    if garak_entries:
        if not g.available():
            raise RuntimeError("Garak is not installed. Install it with:  pip install garak")
        print("Checking probe names ...")
        g.check_probe_names(garak_entries)
    if pyrit_entries and not pe.available(cfg):
        raise RuntimeError(pe.NOT_AVAILABLE.format(python=pe.python_for(cfg)))
    run_dir = Path(cfg["output"]["dir"]) / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    cfg["_run_dir"] = str(run_dir)
    if garak_entries:
        print(f"Garak target: {llm['type']} {llm.get('name', '')}".rstrip())
        print(f"Garak limits: up to {run['max_prompts_per_probe']} prompts per probe x {run['generations']} replies "
              f"each (raise run.max_prompts_per_probe for a fuller scan).")
    if pyrit_entries:
        total = sum(pe.estimate_requests(a) for a in pyrit_entries)
        print(f"PyRIT: about {total} model requests at most across {len(pyrit_entries)} attack entr"
              f"{'y' if len(pyrit_entries) == 1 else 'ies'} (target, attacker and judge calls together).")
    print("Scans send many requests: make sure you are allowed to test this endpoint and expect API costs.\n")

    def run_one(atk, idx=0):
        print("  (a scan can take several minutes; the tool's own log is saved in the results folder)")
        if atk.get("engine") == "pyrit":
            return pe.run_entry(cfg, atk, idx, run_dir / "pyrit", cfg["output"]["save_examples"])
        return g.run_spec(cfg, atk, idx, run_dir / "garak", cfg["output"]["save_examples"])

    versions = {"numpy": np.__version__}
    if garak_entries:
        versions["garak"] = g.version()
    if pyrit_entries:
        versions["pyrit"] = pe.version(cfg)
    return run_one, versions, "n/a"


def _print_probes(filter_text: str) -> int:
    from advassess.engines import garak_engine as g
    if not g.available():
        print("Garak is not installed. Install it with:  pip install garak")
        return 2
    modules = g.list_probes()
    if filter_text and filter_text in modules:
        print(f"Probes in '{filter_text}' (a spec of the module name runs the active ones; * = inactive by default):")
        for name, active in modules[filter_text]:
            print(f"  {name}{'' if active else '  *'}")
        return 0
    print("Garak probe modules (name them under attacks, or add .ClassName for one probe):")
    for m, probes in modules.items():
        if filter_text and filter_text not in m:
            continue
        print(f"  {m:24s}{sum(a for _, a in probes):3d} active of {len(probes)}")
    print("\nList one module's probes:  python run.py --list-probes dan")
    return 0


def _draw(run_dir) -> int:
    """Charts and report.html; never lets a plotting problem break an assessment."""
    try:
        from advassess.visualize import make_visuals
        files = make_visuals(run_dir)
    except ImportError:
        print("\nTo also get charts and report.html, install matplotlib:  pip install matplotlib")
        return 1
    except Exception as exc:
        print(f"\nCould not draw charts: {type(exc).__name__}: {exc}")
        return 1
    print("\nCharts and report saved:")
    for f in files:
        print(f"  {f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adversarial robustness assessment")
    parser.add_argument("--config", help="Path to the YAML config file")
    parser.add_argument("--list-attacks", action="store_true", help="Show available attacks and exit")
    parser.add_argument("--describe-attack", metavar="NAME", help="Show one attack's settings and exit")
    parser.add_argument("--list-probes", nargs="?", const="", metavar="FILTER",
                        help="List Garak probe modules for LLM scans (or one module's probes) and exit")
    parser.add_argument("--visualize", metavar="RUN_FOLDER",
                        help="Draw charts and report.html for an existing results folder, then exit")
    args = parser.parse_args(argv)
    sys.path.insert(0, os.getcwd())  # lets model_class point at a .py file in the current folder
    registry = _registry()

    if args.list_attacks:
        _print_attack_list()
        return 0
    if args.describe_attack:
        return _print_attack_description(args.describe_attack)
    if args.list_probes is not None:
        return _print_probes(args.list_probes)
    if args.visualize:
        return _draw(args.visualize)
    if not args.config:
        parser.error("--config is required")

    try:
        cfg = load_config(args.config, registry)
    except ConfigError as exc:
        print(f"Config problem: {exc}")
        return 2

    np.random.seed(cfg["run"]["seed"])
    kind = target_kind(cfg)
    try:
        run_one, versions, device = {"text": _run_text, "llm": _run_llm}.get(kind, _run_art)(cfg)
    except (FileNotFoundError, KeyError, ValueError, TypeError, RuntimeError, ImportError, AttributeError) as exc:
        print(f"Could not load the model or data: {exc}")
        return 2

    results = []
    for idx, atk in enumerate(cfg["attacks"], start=1):
        label = (f"pyrit.{atk.get('strategy')}" if atk.get("engine") == "pyrit"
                 else atk.get("name") or f"tag:{atk.get('tag')}")
        print(f"Running {label.upper() if kind != 'llm' else label} {atk['params'] or ''} ...")
        try:
            limit = cfg["run"].get("attack_timeout_seconds") or (
                DEFAULT_SUITE_TIMEOUT if cfg.get("_default_attacks") and kind == "image" else None)
            out = _with_timeout(limit, run_one, atk, idx) if kind != "llm" else run_one(atk, idx)
            for result in (out if isinstance(out, list) else [out]):
                if kind == "image" and result["mean_l2"] == 0 and not result["attack_success_rate"]:
                    result["warning"] = (
                        "This attack changed no inputs at all, so the result says nothing about robustness. "
                        "Its settings are probably too weak (try more iterations or a larger step or "
                        "initial_const; see: python run.py --describe-attack " + atk["name"] + ").")
                    print(f"  WARNING: {result['warning']}")
                results.append(result)
        except Exception as exc:  # keep going so one bad attack doesn't lose the others
            if kind == "image" and cfg.get("_default_attacks"):
                reason = f"{type(exc).__name__}: {str(exc)[:140]}"
                print(f"  skipped: {reason}")
                cfg.setdefault("_skipped", []).append({"attack": label.upper(), "params": atk["params"], "reason": reason})
                continue
            if kind != "llm":
                traceback.print_exc(limit=2)
            else:
                print(f"  ERROR: {exc}")
            results.append({"attack": label if kind == "llm" else label.upper(), "params": atk["params"],
                            "error": f"{type(exc).__name__}: {exc}", "examples": []})

    if not results:
        print("\nNo attack produced a result.")
        for sk in cfg.get("_skipped", []):
            print(f"  skipped {sk['attack']}: {sk['reason']}")
        return 1
    print("\n" + format_table(results, kind))
    if cfg.get("_skipped"):
        print("\nSkipped attacks:")
        for sk in cfg["_skipped"]:
            print(f"  {sk['attack']}: {sk['reason']}")
    for r in results:
        if r.get("warning"):
            print(f"\nWARNING ({r['attack']} {r['params']}): {r['warning']}")
    out_dir = write_outputs(cfg, results, versions, device, kind)
    print("\nLimits of this assessment:")
    for line in limits_for(kind):
        print(f"  - {line}")
    print(f"\nReport saved to: {out_dir}")
    _draw(out_dir)
    return 0 if all(not r["error"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

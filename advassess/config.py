"""Load and validate the YAML config that drives an assessment run."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "target": {
        "source": None,            # torchscript | pickle | state_dict | keras | sklearn | llm
        "path": None,
        "input_type": "text",      # sklearn only: text (raw-text pipeline) or tabular (numeric features)
        "outputs": "auto",         # keras only: auto | logits | probabilities
        "model_class": None,       # state_dict only: "package.module:ClassName" or "file.py:ClassName"
        "model_args": {},          # state_dict only: arguments for the class constructor
        "input_shape": None,       # shape of ONE sample, without the batch dimension (default: read from the data)
        "num_classes": None,       # default: read from the model
        "clip_values": None,       # [min, max] allowed input value (default: inferred from the data and printed)
        "device": "auto",          # auto | cpu | cuda
        "llm": {                   # source: llm only (scanned with Garak)
            "type": None,          # Garak generator, e.g. openai, huggingface, ollama, rest
            "name": "",            # model name for that generator
            "options": {},         # generator options, nested as {family: {Class: {option: value}}}
            "api_key_env": None,   # NAME of the environment variable holding the API key
        },
    },
    "data": {
        "source": None,            # npz | csv
        "path": None,              # .npz (x_test, y_test) or .csv file
        "text_column": "text",     # csv only: column with the input text
        "label_column": "label",   # csv only: column with the true label
        "feature_columns": None,   # numeric csv only: columns to use as features (default: all but the label)
        "max_samples": 500,        # cap on test samples, keeps big-model runs affordable
    },
    "run": {
        "batch_size": 64, "seed": 0,
        "generations": 3,              # LLM only: replies requested per prompt
        "max_prompts_per_probe": 64,   # LLM only: cap on prompts per probe (raise for a fuller scan)
        "parallel_requests": 1,        # LLM only
        "timeout_minutes": 60,         # LLM only: limit per attack entry
        "attack_timeout_seconds": None,  # numeric/text models: stop an attack that runs longer (default suite: 180)
    },
    "pyrit": {                     # LLM scans with engine: pyrit
        "objective_target": None,  # {endpoint, model, api_key_env}: the model under test
        "adversarial_target": None,  # attacker LLM for multi-turn strategies
        "scoring_target": None,    # judge LLM, only for llm_judge scoring
        "max_completion_tokens": 400,
        "python": None,            # python executable of the environment where PyRIT is installed
    },
    "attacks": [],                 # leave out (or write "default") to run the built-in default suite
    "output": {"dir": "results", "save_examples": 8},
}

TARGET_SOURCES = {"torchscript", "pickle", "state_dict", "keras", "sklearn", "llm"}
DATA_SOURCES = {"npz", "csv"}


def _wants_default_attacks(cfg: dict) -> bool:
    a = cfg.get("attacks")
    return a is None or a == [] or (isinstance(a, str) and a.strip().lower() == "default")


class ConfigError(Exception):
    """Raised when the config file is missing something or has a bad value."""


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def target_kind(cfg: dict) -> str:
    """'text' (scikit-learn text classifiers), 'llm' (language models) or 'image' (everything attacked
    through ART: PyTorch, Keras and scikit-learn numeric models)."""
    t = cfg["target"]
    if t["source"] == "llm":
        return "llm"
    if t["source"] == "sklearn" and t.get("input_type", "text") == "text":
        return "text"
    return "image"


def load_config(path: str | Path, registry: dict[str, set[str]]) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _merge(DEFAULTS, raw)
    cfg["_config_path"] = str(path.resolve())
    if _wants_default_attacks(cfg):
        cfg["attacks"] = []
        cfg["_default_attacks"] = True
    _validate(cfg, registry)
    return cfg


def _validate(cfg: dict, registry: dict[str, set[str]]) -> None:
    t = cfg["target"]
    if t["source"] not in TARGET_SOURCES:
        raise ConfigError(f"target.source must be one of {sorted(TARGET_SOURCES)}")
    if t["source"] != "llm" and not t.get("path"):
        raise ConfigError("target.path is required (the model file) unless target.source is llm")
    if t["source"] == "state_dict":
        mc = t.get("model_class")
        if not (isinstance(mc, str) and ":" in mc):
            raise ConfigError(
                "target.model_class is required for state_dict, written like "
                "'my_package.models:MyNet' or 'my_model.py:Net'"
            )
        if not isinstance(t.get("model_args"), dict):
            raise ConfigError("target.model_args must be a mapping, e.g. {num_classes: 10}")
    kind = target_kind(cfg)
    if kind == "llm":
        _validate_llm(cfg)
        return
    if t["source"] == "sklearn" and t["input_type"] not in {"text", "tabular"}:
        raise ConfigError("target.input_type must be text or tabular")
    if t["outputs"] not in {"auto", "logits", "probabilities"}:
        raise ConfigError("target.outputs must be auto, logits or probabilities")
    if kind == "image":
        if t["input_shape"] is not None and not (isinstance(t["input_shape"], list)
                                                 and all(isinstance(i, int) for i in t["input_shape"])):
            raise ConfigError("target.input_shape must be a list of integers, e.g. [3, 224, 224]")
        if t["num_classes"] is not None and (not isinstance(t["num_classes"], int) or t["num_classes"] < 2):
            raise ConfigError("target.num_classes must be an integer >= 2")
        cv = t["clip_values"]
        if cv is not None and not (isinstance(cv, list) and len(cv) == 2 and cv[0] < cv[1]):
            raise ConfigError("target.clip_values must be [min, max] with min < max")
        if t["device"] not in {"auto", "cpu", "cuda"}:
            raise ConfigError("target.device must be auto, cpu or cuda")

    d = cfg["data"]
    if d["source"] not in DATA_SOURCES:
        raise ConfigError(f"data.source must be one of {sorted(DATA_SOURCES)}")
    if d["source"] in {"npz", "csv"} and not d.get("path"):
        raise ConfigError(f"data.path is required when data.source is '{d['source']}'")
    if kind == "text" and d["source"] != "csv":
        raise ConfigError("Text classifiers (target.source: sklearn) need data.source: csv")
    if t["source"] == "sklearn" and kind == "image" and d["source"] not in {"csv", "npz"}:
        raise ConfigError("Numeric scikit-learn models need data.source: csv or npz")
    if d.get("feature_columns") is not None and not (isinstance(d["feature_columns"], list)
                                                      and all(isinstance(c, str) for c in d["feature_columns"])):
        raise ConfigError("data.feature_columns must be a list of column names")
    if not isinstance(d["max_samples"], int) or d["max_samples"] < 1:
        raise ConfigError("data.max_samples must be a positive integer")

    if not isinstance(cfg["run"]["batch_size"], int) or cfg["run"]["batch_size"] < 1:
        raise ConfigError("run.batch_size must be a positive integer")
    ats = cfg["run"].get("attack_timeout_seconds")
    if ats is not None and (not isinstance(ats, (int, float)) or ats <= 0):
        raise ConfigError("run.attack_timeout_seconds must be a positive number (or left out)")

    attacks = cfg["attacks"]
    if cfg.get("_default_attacks"):
        return
    known_attacks = registry[kind]
    for i, atk in enumerate(attacks):
        if not isinstance(atk, dict) or "name" not in atk:
            raise ConfigError(f"attacks[{i}] needs a 'name'")
        if atk["name"].lower() not in known_attacks:
            from advassess.engines.art_catalog import UNSUPPORTED

            reason = {k.lower(): v for k, v in UNSUPPORTED.items()}.get(atk["name"].lower())
            if reason and kind == "image":
                raise ConfigError(f"Attack '{atk['name']}' is not supported by this tool: {reason}.")
            raise ConfigError(
                f"Unknown attack '{atk['name']}'. See the available attacks with: "
                "python run.py --list-attacks"
            )
        atk.setdefault("params", {})
        if atk["params"] is None:
            atk["params"] = {}
        if "samples" in atk and (not isinstance(atk["samples"], int) or atk["samples"] < 1):
            raise ConfigError(f"attacks[{i}].samples must be a positive integer")


def _validate_llm(cfg: dict) -> None:
    from advassess.engines import garak_engine as g
    from advassess.engines import pyrit_engine as pe

    llm = cfg["target"]["llm"]
    attacks = cfg["attacks"]
    default_run = bool(cfg.get("_default_attacks"))
    uses_garak = default_run or any(isinstance(a, dict) and a.get("engine") != "pyrit" for a in attacks)

    if uses_garak:
        if not (isinstance(llm.get("type"), str) and g.GENERATOR_NAME.match(llm["type"])):
            raise ConfigError("target.llm.type is required: the Garak generator, e.g. openai, huggingface, "
                              "ollama or rest (see: python -m garak --list_generators)")
    if llm.get("name") is not None and not isinstance(llm["name"], (str, int)):
        raise ConfigError("target.llm.name must be text")
    if not isinstance(llm.get("options"), dict):
        raise ConfigError("target.llm.options must be a mapping like {openai: {OpenAICompatible: {uri: ...}}}")
    secrets = g.find_secret_keys(llm["options"])
    if secrets:
        raise ConfigError(
            f"target.llm.options contains what looks like a secret ({', '.join(secrets)}). Do not put keys in "
            "the config: set the key in an environment variable and name it in target.llm.api_key_env.")
    env = llm.get("api_key_env")
    if env is not None and not (isinstance(env, str) and env.isidentifier()):
        raise ConfigError("target.llm.api_key_env must be an environment variable NAME such as OPENAI_API_KEY")

    run = cfg["run"]
    for key in ("generations", "max_prompts_per_probe", "parallel_requests"):
        if not isinstance(run[key], int) or run[key] < 1:
            raise ConfigError(f"run.{key} must be a positive integer")
    if not isinstance(run["timeout_minutes"], (int, float)) or run["timeout_minutes"] <= 0:
        raise ConfigError("run.timeout_minutes must be a positive number")
    if not isinstance(cfg["pyrit"]["max_completion_tokens"], int) or cfg["pyrit"]["max_completion_tokens"] < 1:
        raise ConfigError("pyrit.max_completion_tokens must be a positive integer")

    if default_run:
        return
    for i, atk in enumerate(attacks):
        if not isinstance(atk, dict):
            raise ConfigError(f"attacks[{i}] must be a mapping")
        if atk.get("engine") == "pyrit":
            pe.validate_entry(i, atk, cfg)
            atk["params"] = {}
            continue
        if atk.get("engine") not in (None, "garak"):
            raise ConfigError(f"attacks[{i}].engine must be garak or pyrit")
        if ("name" in atk) == ("tag" in atk):
            raise ConfigError(f"attacks[{i}] needs exactly one of 'name' (a Garak probe) or 'tag'")
        if "name" in atk and not g.PROBE_NAME.match(str(atk["name"])):
            raise ConfigError(f"attacks[{i}].name must look like 'dan' or 'dan.Dan_11_0' (see: python run.py --list-probes)")
        if "tag" in atk and not g.TAG_NAME.match(str(atk["tag"])):
            raise ConfigError(f"attacks[{i}].tag has unusual characters; use something like owasp:llm01")
        if atk.get("params"):
            raise ConfigError(f"attacks[{i}].params is not supported for LLM probes yet; use generations, "
                              "max_prompts_per_probe or samples to control a scan")
        if "samples" in atk and (not isinstance(atk["samples"], int) or atk["samples"] < 1):
            raise ConfigError(f"attacks[{i}].samples must be a positive integer")
        atk["params"] = {}

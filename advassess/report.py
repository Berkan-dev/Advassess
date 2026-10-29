"""Turn attack results into a console table, JSON, CSV and an audit-log entry."""
from __future__ import annotations

import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LIMITS = [
    "Results apply only to the attacks, parameters and samples listed above.",
    "A low attack success rate does not prove robustness: stronger attacks, "
    "different parameters or black-box methods may still succeed.",
    "These are white-box attacks: they use the model's gradients, which an outside "
    "attacker may not have.",
    "Attack success rate counts only samples the model classified correctly before the attack.",
]


TEXT_LIMITS = [
    "Results apply only to the attacks, parameters and samples listed above.",
    "A low attack success rate does not prove robustness: stronger or smarter attacks may still succeed.",
    "These attacks only query the model (black-box), so they are a floor on weakness, not a ceiling.",
    "Word-insertion candidates are taken from the test data, so a small or unrepresentative test set weakens the test.",
    "Attack success rate counts only samples the model classified correctly before the attack.",
    "Perturbed texts may not read naturally; no check is made that a human would read them the same way.",
]


LLM_LIMITS = [
    "Results apply only to the probes, prompt caps and number of replies listed above.",
    "Passing a probe is not a guarantee of safety: Garak covers a fixed set of known attack patterns.",
    "A 'fail' means the reply matched an automated detector (for example, it followed an injected "
    "instruction). Detectors make mistakes in both directions, so read the examples before concluding anything.",
    "Language models answer differently from run to run; rates from few prompts or few replies are noisy. "
    "Repeat the scan or raise generations / max_prompts_per_probe before relying on a rate.",
    "Prompts per probe are capped (see max_prompts_per_probe and samples), so a probe may be only partly run.",
    "Probes are single prompts or scripted conversations; a determined human or an adaptive attacker may do better.",
    "Adaptive (PyRIT) attacks depend on the attacker and judge models you choose: a weak attacker underestimates "
    "the risk, and a judge can mislabel a transcript. Read the transcripts.",
]


def limits_for(kind: str) -> list[str]:
    return {"text": TEXT_LIMITS, "llm": LLM_LIMITS}.get(kind, LIMITS)


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def _params(p: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items()) or "(defaults)"


def _format_llm_table(results: list[dict]) -> str:
    headers = ["Probe", "Prompts", "Replies", "Failed", "Attack success", "Detectors", "Run sec"]
    rows, errors = [], []
    for r in results:
        if r["error"]:
            rows.append([r["attack"], "ERROR", "see below", "", "", "", ""])
            errors.append(f"  {r['attack']}: {r['error']}")
            continue
        rows.append([r["attack"], r["n_prompts"], r["n_samples"], r["failed"], _pct(r["attack_success_rate"]),
                     ", ".join(r["detectors"]) or "-", r["seconds"]])
    widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows)]
    line = lambda cells: "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))
    table = "\n".join([line(headers), line(["-" * w for w in widths])] + [line(r) for r in rows])
    return table + ("\n\nErrors:\n" + "\n".join(errors) if errors else "")


def format_table(results: list[dict], kind: str = "image") -> str:
    if kind == "llm":
        return _format_llm_table(results)
    extra_headers = ["Words changed", "Queries"] if kind == "text" else ["Mean L2", "Mean Linf"]
    headers = ["Attack", "Parameters", "Clean acc", "Adv acc", "Success"] + extra_headers + ["Sec"]
    rows = []
    for r in results:
        if r["error"]:
            rows.append([r["attack"], _params(r["params"]), "ERROR", r["error"][:60], "", "", "", ""])
            continue
        extra = ([f"{r['mean_words_changed']:.2f}", f"{r['mean_queries']:.0f}"] if kind == "text"
                 else [f"{r['mean_l2']:.3f}", f"{r['mean_linf']:.3f}"])
        rows.append([r["attack"], _params(r["params"]), _pct(r["clean_accuracy"]),
                     _pct(r["adversarial_accuracy"]), _pct(r["attack_success_rate"])] + extra + [f"{r['seconds']}"])
    widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows)]
    line = lambda cells: "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))
    return "\n".join([line(headers), line(["-" * w for w in widths])] + [line(r) for r in rows])


def write_outputs(cfg: dict, results: list[dict], versions: dict, device: str, kind: str = "image") -> Path:
    stamp = datetime.now(timezone.utc)
    run_dir = Path(cfg.get("_run_dir") or Path(cfg["output"]["dir"]) / stamp.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    clean_results = []
    for idx, r in enumerate(results, start=1):
        r = dict(r)
        examples = r.pop("examples", [])
        stem = f"examples_{idx:02d}_{r['attack'].lower()}"
        if examples:
            r["examples_file"] = stem + (".json" if kind in ("text", "llm") else ".npz")
        if examples and kind == "llm":
            (run_dir / f"{stem}.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
        elif examples and kind == "text":
            (run_dir / f"{stem}.json").write_text(
                json.dumps([{"original": e[0], "adversarial": e[1], "true_label": e[2],
                             "adversarial_label": e[3]} for e in examples], indent=2, ensure_ascii=False),
                encoding="utf-8")
        elif examples:
            np.savez_compressed(
                run_dir / f"{stem}.npz",
                original=np.stack([e[0] for e in examples]),
                adversarial=np.stack([e[1] for e in examples]),
                true_label=np.array([e[2] for e in examples]),
                adversarial_label=np.array([e[3] for e in examples]),
            )
        clean_results.append(r)

    t = cfg["target"]
    report = {
        "timestamp_utc": stamp.isoformat(),
        "target": ({"source": "llm", "kind": "llm", "generator": t["llm"]["type"] or "pyrit",
                    "model": t["llm"].get("name") or ((cfg.get("pyrit") or {}).get("objective_target") or {}).get("model", ""),
                    "options": t["llm"].get("options", {})} if kind == "llm" else
                   {"source": t["source"], "path": t["path"], "kind": "text"} if kind == "text" else
                   {"source": t["source"], "path": t["path"], "input_shape": t["input_shape"],
                    "num_classes": t["num_classes"], "clip_values": t["clip_values"], "device": device,
                    "kind": "image"}),
        "data": ({"source": "none (LLM probes)"} if kind == "llm" else
                 {"source": cfg["data"]["source"], "path": cfg["data"]["path"],
                  "max_samples": cfg["data"]["max_samples"]}),
        "run": cfg["run"],
        "default_attack_suite": bool(cfg.get("_default_attacks")),
        "skipped_attacks": cfg.get("_skipped", []),
        "environment": {"python": platform.python_version(), **versions},
        "results": clean_results,
        "limits": limits_for(kind),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    with open(run_dir / "report.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["attack", "params", "n_samples", "clean_accuracy", "adversarial_accuracy",
                    "attack_success_rate", "mean_l2", "mean_linf", "mean_words_changed", "mean_queries",
                    "n_prompts", "failed", "seconds", "error"])
        for r in clean_results:
            w.writerow([r["attack"], json.dumps(r["params"]), r.get("n_samples"),
                        r.get("clean_accuracy"), r.get("adversarial_accuracy"),
                        r.get("attack_success_rate"), r.get("mean_l2"), r.get("mean_linf"),
                        r.get("mean_words_changed"), r.get("mean_queries"),
                        r.get("n_prompts"), r.get("failed"), r.get("seconds"), r.get("error")])

    # Append-only audit log: who/what/when for every run.
    log_path = Path(cfg["output"]["dir"]) / "run_log.jsonl"
    entry = {"timestamp_utc": stamp.isoformat(), "report": str(run_dir / "report.json"),
             "target": (f"llm:{t['llm']['type'] or 'pyrit'}:{t['llm'].get('name', '')}" if kind == "llm"
                        else t["path"]),
             "attacks": [r["attack"] for r in clean_results]}
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return run_dir

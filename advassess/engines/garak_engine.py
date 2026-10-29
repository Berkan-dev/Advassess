"""LLM scanning with NVIDIA's Garak.

Garak is run as a separate process (python -m garak ...), one run per entry under
`attacks`, so it stays an optional dependency and a Garak crash or hang cannot take
the assessment down. Its report.jsonl is then read back into the same kind of result
rows the other engines produce.

An attack entry is a Garak probe: a module ("dan", "encoding") runs all of that
module's default probes, "module.Class" runs one, and `tag: owasp:llm01` selects by tag.
"""
from __future__ import annotations

import difflib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_SECRET_WORDS = ("key", "token", "secret", "password", "passwd", "credential", "authorization")
PROBE_NAME = re.compile(r"^[a-z0-9_]+(\.[A-Za-z0-9_]+)?$")
TAG_NAME = re.compile(r"^[A-Za-z0-9_:.\-]+$")
GENERATOR_NAME = re.compile(r"^[A-Za-z0-9_.]+$")
HIT_THRESHOLD = 0.5  # Garak's own default: a detector score above this counts as a hit


def available() -> bool:
    return importlib.util.find_spec("garak") is not None


def version() -> str:
    try:
        from importlib.metadata import version as pkg_version
        return pkg_version("garak")
    except Exception:
        return "unknown"


def _garak(*args: str) -> list[str]:
    return [sys.executable, "-m", "garak", *args]


def find_secret_keys(options, path: str = "") -> list[str]:
    """Names of option keys that look like they hold a secret."""
    found = []
    if isinstance(options, dict):
        for k, v in options.items():
            here = f"{path}.{k}" if path else str(k)
            if any(w in str(k).lower() for w in _SECRET_WORDS):
                found.append(here)
            found += find_secret_keys(v, here)
    elif isinstance(options, list):
        for i, v in enumerate(options):
            found += find_secret_keys(v, f"{path}[{i}]")
    return found


def list_probes() -> dict[str, list[tuple[str, bool]]]:
    """{module: [(probe, active_by_default), ...]} read from `garak --list_probes`."""
    out = subprocess.run(_garak("--list_probes"), capture_output=True, text=True, timeout=180)
    modules: dict[str, list[tuple[str, bool]]] = {}
    for line in _ANSI.sub("", out.stdout).splitlines():
        if not line.startswith("probes: "):
            continue
        name = line.split()[1]
        if "." in name:
            modules.setdefault(name.split(".")[0], []).append((name, "💤" not in line))
        else:
            modules.setdefault(name, [])
    return modules


def check_probe_names(attacks: list[dict]) -> None:
    """Fail early, with suggestions, if a probe name does not exist."""
    wanted = [a["name"] for a in attacks if "name" in a]
    if not wanted:
        return
    modules = list_probes()
    known = set(modules) | {p for plist in modules.values() for p, _ in plist}
    for name in wanted:
        if name not in known:
            close = difflib.get_close_matches(name, sorted(known), n=4)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise ValueError(f"Garak has no probe '{name}'.{hint} List them with: python run.py --list-probes")


def _spec(atk: dict) -> str:
    return f"tag:{atk['tag']}" if "tag" in atk else f"probes.{atk['name']}"


def _tail(text: str, n: int = 6) -> str:
    lines = [l for l in _ANSI.sub("", text).splitlines() if l.strip()]
    return " | ".join(lines[-n:])[:600]


def _clip(text: str, n: int = 600) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + " ... [cut]"


def _turns_text(prompt: dict) -> str:
    turns = (prompt or {}).get("turns") or []
    texts = [(t.get("role", "user"), (t.get("content") or {}).get("text", "")) for t in turns]
    if len(texts) == 1:
        return texts[0][1]
    return "\n".join(f"[{role}] {text}" for role, text in texts)


def parse_report(report_path: Path, seconds: float, spec: str, n_examples: int) -> list[dict]:
    """One result row per probe found in a Garak report.jsonl."""
    probes: dict[str, dict] = {}
    evals: dict[str, dict[str, dict]] = {}
    for line in report_path.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = d.get("entry_type")
        if kind == "eval":
            evals.setdefault(d["probe"], {})[d["detector"]] = {
                "passed": d.get("passed", 0), "failed": d.get("fails", 0),
                "evaluated": d.get("total_evaluated", 0)}
        elif kind == "attempt" and d.get("status") == 2:  # status 2 = detectors have scored it
            p = probes.setdefault(d["probe_classname"], {"prompts": 0, "outputs": 0, "hits": 0, "examples": []})
            p["prompts"] += 1
            scores = d.get("detector_results") or {}
            width = max((len(v) for v in scores.values()), default=0)
            for i in range(width):
                vals = {det: s[i] for det, s in scores.items() if i < len(s) and s[i] is not None}
                if not vals:
                    continue
                p["outputs"] += 1
                fired = sorted(det for det, s in vals.items() if s >= HIT_THRESHOLD)
                if fired:
                    p["hits"] += 1
                    if len(p["examples"]) < n_examples and i < len(d.get("outputs") or []):
                        p["examples"].append({
                            "prompt": _clip(_turns_text(d.get("prompt"))),
                            "response": _clip((d["outputs"][i] or {}).get("text", "")),
                            "detectors": fired,
                            "goal": d.get("goal") or ""})
    rows = []
    for name, p in probes.items():
        rows.append({
            "attack": name,
            "params": {"spec": spec},
            "n_prompts": p["prompts"],
            "n_samples": p["outputs"],
            "failed": p["hits"],
            "attack_success_rate": (p["hits"] / p["outputs"]) if p["outputs"] else None,
            "detectors": evals.get(name, {}),
            "seconds": round(seconds, 1),
            "examples": p["examples"],
            "error": None,
        })
    return rows


def run_spec(cfg: dict, atk: dict, index: int, work_dir: Path, n_examples: int) -> list[dict]:
    """Run Garak for one attack entry and return its result rows."""
    llm = cfg["target"]["llm"]
    run = cfg["run"]
    env_var = llm.get("api_key_env")
    if env_var and not os.environ.get(env_var):
        raise RuntimeError(f"Environment variable {env_var} is not set. Set it to your API key "
                           "before running; the key is never written to the config or reports.")

    work_dir = Path(work_dir).resolve()  # Garak resolves relative report paths inside its own folder
    work_dir.mkdir(parents=True, exist_ok=True)
    spec = _spec(atk)
    stem = f"garak_{index:02d}_{re.sub(r'[^A-Za-z0-9]+', '_', spec).strip('_')}"
    prefix = work_dir / stem

    cap = atk.get("samples") or run["max_prompts_per_probe"]
    cfg_file = work_dir / f"{stem}.config.json"
    cfg_file.write_text(json.dumps({"run": {"soft_probe_prompt_cap": int(cap)}}), encoding="utf-8")

    cmd = _garak("--target_type", llm["type"])
    if llm.get("name"):
        cmd += ["--target_name", str(llm["name"])]
    cmd += ["--spec", spec, "--generations", str(run["generations"]), "--seed", str(run["seed"]),
            "--report_prefix", str(prefix), "--config", str(cfg_file)]
    if llm.get("options"):
        opt_file = work_dir / f"{stem}.generator_options.json"
        opt_file.write_text(json.dumps(llm["options"]), encoding="utf-8")
        cmd += ["--generator_option_file", str(opt_file)]
    if run.get("parallel_requests", 1) > 1:
        cmd += ["--parallel_requests", str(run["parallel_requests"])]
    started = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy(),
                              timeout=float(run["timeout_minutes"]) * 60)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Garak did not finish within {run['timeout_minutes']} minutes and was stopped "
                           "(raise run.timeout_minutes, lower generations, or use samples)") from None
    seconds = time.perf_counter() - started
    (work_dir / f"{stem}.log").write_text(_ANSI.sub("", proc.stdout + "\n" + proc.stderr), encoding="utf-8")

    report = Path(str(prefix) + ".report.jsonl")
    crashed = proc.returncode != 0
    rows = parse_report(report, seconds, spec, n_examples) if report.is_file() else []
    if crashed:
        rows = [r for r in rows if r["detectors"]]  # a probe cut off mid-way has no verdict yet
        reason = f"Garak stopped early (exit {proc.returncode}): {_tail(proc.stdout + proc.stderr)}"
        if not rows:
            raise RuntimeError(reason)
        rows.append({"attack": spec, "params": {"spec": spec}, "error": reason + f" | {len(rows)} probe(s) "
                     f"finished before this; probes after the failure were not run. Log: {stem}.log",
                     "examples": []})
    elif not rows:
        raise RuntimeError("Garak finished but scored no prompts. See " + str(work_dir / f"{stem}.log"))
    return rows

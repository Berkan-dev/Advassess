"""Charts and an HTML report built from a finished run folder.

Reads report.json and the saved examples_* files, so it works right after a run
and on any older results folder:

    python run.py --visualize results/20261024-134739

Writes into the run folder:
    summary.png          clean vs attacked accuracy for every attack
    sweep.png            accuracy vs attack strength (only if the same attack was run at 2+ strengths)
    examples_<attack>.png  original / adversarial / change, for image models
    report.html          everything above in one self-contained page
"""
from __future__ import annotations

import base64
import difflib
import html
import io
import json
from pathlib import Path

import numpy as np

from advassess.report import _params, _pct

CLEAN, ATTACKED, CHANGE = "#4C78A8", "#D1495B", "#7A7A7A"


def _plt():
    import matplotlib

    matplotlib.use("Agg")  # draw to files only, no window needed
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _label(r: dict) -> str:
    return f"{r['attack']}  {_params(r['params'])}"


def _png_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


# ----------------------------------------------------------------- charts

def _summary_chart(results: list[dict], out: Path) -> Path:
    plt = _plt()
    n = len(results)
    fig, ax = plt.subplots(figsize=(9.5, 2.3 + 0.8 * n))
    ys = np.arange(n)
    h = 0.38
    clean = [r["clean_accuracy"] * 100 for r in results]
    adv = [r["adversarial_accuracy"] * 100 for r in results]
    ax.barh(ys - h / 2, clean, h, color=CLEAN, label="Clean accuracy")
    ax.barh(ys + h / 2, adv, h, color=ATTACKED, label="Accuracy under attack")
    for y, c, a, r in zip(ys, clean, adv, results):
        ax.text(c + 1, y - h / 2, f"{c:.1f}%", va="center", fontsize=9)
        rate = r["attack_success_rate"]
        tail = f"   (fooled {rate * 100:.1f}% of correct)" if rate is not None else ""
        ax.text(a + 1, y + h / 2, f"{a:.1f}%{tail}", va="center", fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels([_label(r) for r in results])
    ax.invert_yaxis()
    ax.set_xlim(0, 135)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Accuracy (%)")
    fig.suptitle("Accuracy before and after each attack", x=0.02, ha="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _llm_chart(results: list[dict], out: Path) -> Path:
    """Share of replies the detectors flagged, per probe."""
    plt = _plt()
    rows = sorted(results, key=lambda r: (r["attack_success_rate"] or 0), reverse=True)
    fig, ax = plt.subplots(figsize=(9.5, 1.4 + 0.45 * len(rows)))
    ys = np.arange(len(rows))
    rates = [(r["attack_success_rate"] or 0) * 100 for r in rows]
    ax.barh(ys, rates, 0.6, color=ATTACKED)
    for y, rate, r in zip(ys, rates, rows):
        ax.text(rate + 1, y, f"{rate:.1f}%  ({r['failed']} of {r['n_samples']} replies)", va="center", fontsize=9)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["attack"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 135)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Replies flagged by the detectors (%)")
    fig.suptitle("How often each probe got a flagged reply", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _sweep_chart(results: list[dict], out: Path) -> Path | None:
    """Accuracy vs eps, one line per attack (same other settings)."""
    groups: dict[str, list[tuple[float, float]]] = {}
    for r in results:
        if "eps" in r["params"]:
            rest = {k: v for k, v in r["params"].items() if k != "eps"}
            key = f"{r['attack']}  {_params(rest)}" if rest else r["attack"]
            groups.setdefault(key, []).append((float(r["params"]["eps"]), r["adversarial_accuracy"] * 100))
    groups = {k: sorted(v) for k, v in groups.items() if len({e for e, _ in v}) >= 2}
    if not groups:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    clean = results[0]["clean_accuracy"] * 100
    ax.axhline(clean, color=CLEAN, ls="--", lw=1)
    ax.text(0.01, clean + 1.5, f"clean accuracy {clean:.1f}%", color=CLEAN, fontsize=9,
            transform=ax.get_yaxis_transform())  # x in axes fraction, y in data units
    for key, pts in groups.items():
        ax.plot([e for e, _ in pts], [a for _, a in pts], marker="o", label=key)
    ax.set_ylim(0, 108)
    ax.set_xlabel("Attack strength (eps)")
    ax.set_ylabel("Accuracy under attack (%)")
    ax.set_title("How accuracy falls as the attack gets stronger", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _as_picture(x: np.ndarray):
    """Turn one sample into something imshow can draw, or None if it isn't image-like."""
    if x.ndim == 3 and x.shape[0] in (1, 3):
        pic = x[0] if x.shape[0] == 1 else np.transpose(x, (1, 2, 0))
    elif x.ndim == 2:
        pic = x
    else:
        return None
    lo, hi = float(pic.min()), float(pic.max())
    return pic if lo >= 0 and hi <= 1 else (pic - lo) / (hi - lo + 1e-9)


def _examples_chart(npz_path: Path, title: str, out: Path) -> Path | None:
    with np.load(npz_path) as d:
        orig, adv = d["original"], d["adversarial"]
        true, pred = d["true_label"], d["adversarial_label"]
    if _as_picture(orig[0]) is None:
        return None
    plt = _plt()
    k = len(orig)
    fig, axes = plt.subplots(3, k, figsize=(1.7 * k + 1.2, 5.6), squeeze=False)
    rows = ["Original", "Attacked", "Change\n(amplified)"]
    for j in range(k):
        o, a = _as_picture(orig[j]), _as_picture(adv[j])
        diff = adv[j] - orig[j]
        diff = diff[0] if diff.ndim == 3 and diff.shape[0] == 1 else (
            np.transpose(diff, (1, 2, 0)).mean(axis=2) if diff.ndim == 3 else diff)
        m = float(np.abs(diff).max()) or 1.0
        cmap = "gray" if o.ndim == 2 else None
        axes[0][j].imshow(o, cmap=cmap, vmin=0, vmax=1)
        axes[1][j].imshow(a, cmap=cmap, vmin=0, vmax=1)
        axes[2][j].imshow(diff, cmap="RdBu_r", vmin=-m, vmax=m)
        axes[0][j].set_title(f"true: {true[j]}", fontsize=9)
        axes[1][j].set_title(f"model says: {pred[j]}", fontsize=9, color=ATTACKED)
        for i in range(3):
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
            for s in axes[i][j].spines.values():
                s.set_visible(False)
    for i, name in enumerate(rows):
        axes[i][0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle(f"Successful attacks: {title}", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ----------------------------------------------------------------- HTML

def _marked(tokens: list[str], keep: set[int]) -> str:
    return " ".join(f"<mark>{html.escape(t)}</mark>" if i in keep else html.escape(t) for i, t in enumerate(tokens))


def _llm_example(e: dict) -> str:
    return (
        f"<div class='ex'><div class='lbl'>Prompt sent"
        + (f" (goal: {html.escape(e['goal'])})" if e.get("goal") else "")
        + f"</div><pre>{html.escape(e['prompt'])}</pre>"
        f"<div class='lbl'>Reply, flagged by <b class='bad'>{html.escape(', '.join(e['detectors']))}</b></div>"
        f"<pre>{html.escape(e['response'])}</pre></div>"
    )


def _text_example(e: dict) -> str:
    a, b = e["original"].split(), e["adversarial"].split()
    changed_a, changed_b = set(), set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag != "equal":
            changed_a.update(range(i1, i2))
            changed_b.update(range(j1, j2))
    return (
        f"<div class='ex'><div class='lbl'>Original, model says "
        f"<b>{html.escape(str(e['true_label']))}</b> (correct)</div><p>{_marked(a, changed_a)}</p>"
        f"<div class='lbl'>Changed, model says <b class='bad'>{html.escape(str(e['adversarial_label']))}</b></div>"
        f"<p>{_marked(b, changed_b)}</p></div>"
    )


CSS = """
:root{color-scheme:light dark;--bg:#fff;--fg:#1d2330;--mut:#667;--line:#dde1e8;--card:#f6f8fb;--bad:#D1495B;--mark:#ffe08a}
@media (prefers-color-scheme:dark){:root{--bg:#14171d;--fg:#e6e9ef;--mut:#9aa3b2;--line:#2c323d;--card:#1c2028;--mark:#6b5a12}}
body{font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:980px;margin:2rem auto;padding:0 1rem}
h1{margin-bottom:.2rem}h2{margin-top:2.2rem;border-bottom:1px solid var(--line);padding-bottom:.3rem}
.meta{color:var(--mut)}table{border-collapse:collapse;width:100%;display:block;overflow-x:auto}
th,td{padding:.45rem .7rem;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--mut);font-weight:600}img{max-width:100%;height:auto;background:#fff;border-radius:6px}
.ex{background:var(--card);border-radius:8px;padding:.7rem 1rem;margin:.8rem 0}.lbl{color:var(--mut);font-size:.85rem}
pre{white-space:pre-wrap;word-break:break-word;margin:.3rem 0 .7rem;font:13px/1.45 ui-monospace,monospace}
mark{background:var(--mark);color:inherit;padding:0 .15rem;border-radius:3px}.bad{color:var(--bad)}li{margin:.25rem 0}
"""


def _html_report(report: dict, run_dir: Path, images: dict[str, Path], text_examples: dict[str, list]) -> Path:
    kind = report["target"].get("kind", "image")
    if kind == "llm":
        head = ["Probe", "Prompts", "Replies", "Failed", "Attack success", "Detectors", "Run sec"]
        rows = []
        for r in report["results"]:
            if r.get("error"):
                rows.append([r["attack"], "ERROR", html.escape(r["error"]), "", "", "", ""])
            else:
                rows.append([r["attack"], r["n_prompts"], r["n_samples"], r["failed"], _pct(r["attack_success_rate"]),
                             ", ".join(r["detectors"]) or "-", r["seconds"]])
    else:
        extra = ["Words changed", "Queries"] if kind == "text" else ["Mean L2", "Mean Linf"]
        head = ["Attack", "Settings", "Clean acc", "Under attack", "Fooled"] + extra + ["Seconds"]
        rows = []
        for r in report["results"]:
            if r.get("error"):
                rows.append([r["attack"], _params(r["params"]), "ERROR", html.escape(r["error"]), "", "", "", ""])
                continue
            ex = ([f"{r['mean_words_changed']:.2f}", f"{r['mean_queries']:.0f}"] if kind == "text"
                  else [f"{r['mean_l2']:.3f}", f"{r['mean_linf']:.3f}"])
            rows.append([r["attack"], _params(r["params"]), _pct(r["clean_accuracy"]), _pct(r["adversarial_accuracy"]),
                         _pct(r["attack_success_rate"])] + ex + [str(r["seconds"])])
    table = ("<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>"
             + "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
                       for row in rows) + "</table>")
    t = report["target"]
    if kind == "llm":
        what, size = f"{t['generator']} {t.get('model', '')}".strip(), ""
    else:
        what, size = str(t.get("path") or t["source"]), f" · {report['data'].get('max_samples')} samples max"
    body = [f"<h1>Adversarial assessment</h1><div class='meta'>{html.escape(report['timestamp_utc'])} · "
            f"target: {html.escape(what)}{size} · "
            f"authorization note: {html.escape(str(report['authorization'].get('note', '')))}</div>",
            "<h2>Results</h2>", table]
    if report.get("default_attack_suite"):
        body.append("<p class='meta'>No attacks were listed in the config, so the built-in default suite ran.</p>")
    for sk in report.get("skipped_attacks", []):
        body.append(f"<p class='meta'>Skipped {html.escape(sk['attack'])} {html.escape(_params(sk['params']))}: "
                    f"{html.escape(sk['reason'])}</p>")
    for r in report["results"]:
        if r.get("warning"):
            body.append(f"<p class='ex'><b class='bad'>Warning</b> ({html.escape(r['attack'])} "
                        f"{html.escape(_params(r['params']))}): {html.escape(r['warning'])}</p>")
    for name in ("summary.png", "sweep.png"):
        if name in images:
            body.append(f"<p><img alt='{name}' src='data:image/png;base64,{_png_b64(images[name])}'></p>")
    ex_keys = [k for k in images if k.startswith("examples_")]
    if ex_keys or text_examples:
        body.append("<h2>Example attacks that succeeded</h2>")
    for k in ex_keys:
        body.append(f"<p><img alt='{k}' src='data:image/png;base64,{_png_b64(images[k])}'></p>")
    render = _llm_example if kind == "llm" else _text_example
    for attack, items in text_examples.items():
        body.append(f"<h3>{html.escape(attack)}</h3>" + "".join(render(e) for e in items))
    body.append("<h2>Limits of this assessment</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in report["limits"]) + "</ul>")
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>Adversarial assessment</title>"
            f"<style>{CSS}</style></head><body>{''.join(body)}</body></html>")
    out = run_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    return out


# ----------------------------------------------------------------- entry point

def make_visuals(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    report_path = run_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"No report.json in {run_dir}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = [r for r in report["results"] if not r.get("error")]
    kind = report["target"].get("kind", "image")

    images: dict[str, Path] = {}
    text_examples: dict[str, list] = {}
    if results and kind == "llm":
        images["summary.png"] = _llm_chart(results, run_dir / "summary.png")
    elif results:
        images["summary.png"] = _summary_chart(results, run_dir / "summary.png")
        sweep = _sweep_chart(results, run_dir / "sweep.png")
        if sweep:
            images["sweep.png"] = sweep
    has_links = any(r.get("examples_file") for r in report["results"])
    if has_links:
        sources = [(r["attack"] if kind == "llm" else _label(r), run_dir / r["examples_file"])
                   for r in results if r.get("examples_file")]
    else:  # older results folders: one examples file per attack name
        sources = [(p.stem[len("examples_"):].upper(), p) for p in sorted(run_dir.glob("examples_*"))]
    for title, path in sources:
        if not path.is_file():
            continue
        if kind in ("text", "llm") and path.suffix == ".json":
            text_examples[title] = json.loads(path.read_text(encoding="utf-8"))
        elif kind == "image" and path.suffix == ".npz":
            made = _examples_chart(path, title, run_dir / f"{path.stem}.png")
            if made:
                images[made.name] = made
    html_path = _html_report(report, run_dir, images, text_examples)
    return [*images.values(), html_path]

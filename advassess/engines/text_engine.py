"""Black-box evasion attacks for text classifiers.

These attacks only need to ask the model for predictions, so they also work on
models that expose no gradients (Naive Bayes, trees, hosted APIs).

char_perturb    Misspell the words the model relies on most (swap letters, drop a
                letter, leet-style digits), a few words at a time, until the
                prediction flips or the word budget runs out.
word_insertion  Append words typical of other classes, one at a time, picking the
                word that lowers the model's confidence the most.

Candidate words for word_insertion come from the test data itself, so the data
should be representative of what the model sees in use.
"""
from __future__ import annotations

import re
import time
from collections import Counter

import numpy as np

ALLOWED_PARAMS = {
    "char_perturb": {"max_words"},
    "word_insertion": {"max_words", "candidates"},
}
DEFAULTS = {
    "char_perturb": {"max_words": 3},
    "word_insertion": {"max_words": 5, "candidates": 50},
}
ATTACKS = set(ALLOWED_PARAMS)
_LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}
_WORD = re.compile(r"[a-z]{3,}")


def known_attacks() -> set[str]:
    return set(ATTACKS)


class _Oracle:
    """Wraps the model and counts how many predictions the attacker asked for."""

    def __init__(self, clf) -> None:
        self.clf = clf
        self.queries = 0

    def p_true(self, texts: list[str], label) -> np.ndarray:
        self.queries += len(texts)
        if self.clf.has_proba:
            return self.clf.predict_proba(texts)[:, self.clf.classes_.index(label)]
        return (self.clf.predict(texts) == label).astype(float)  # coarse fallback: 1 if still correct

    def label(self, text: str):
        self.queries += 1
        return self.clf.predict([text])[0]


def _char_variants(token: str) -> list[str]:
    out = set()
    n = len(token)
    for i in range(1, n - 2):                       # swap two neighbouring inner letters
        out.add(token[:i] + token[i + 1] + token[i] + token[i + 2:])
    for i in range(1, n - 1):                       # drop one inner letter
        out.add(token[:i] + token[i + 1:])
    for i, ch in enumerate(token):                  # leet-style substitution
        if ch.lower() in _LEET:
            out.add(token[:i] + _LEET[ch.lower()] + token[i + 1:])
    out.discard(token)
    return sorted(out)


def _attack_char(text: str, label, oracle: _Oracle, max_words: int):
    tokens = text.split()
    if not tokens:
        return text, 0, False
    cur_p = oracle.p_true([text], label)[0]
    without = [" ".join(tokens[:i] + tokens[i + 1:]) for i in range(len(tokens))]
    importance = cur_p - oracle.p_true(without, label)   # how much each word props up the right answer
    current, changed = tokens[:], 0
    for i in np.argsort(-importance):
        if changed >= max_words:
            break
        variants = _char_variants(current[i])
        if not variants:
            continue
        texts = [" ".join(current[:i] + [v] + current[i + 1:]) for v in variants]
        p = oracle.p_true(texts, label)
        j = int(np.argmin(p))
        if p[j] >= cur_p:
            continue
        current[i], cur_p, changed = variants[j], float(p[j]), changed + 1
        if oracle.label(" ".join(current)) != label:
            return " ".join(current), changed, True
    return " ".join(current), changed, False


def _attack_insert(text: str, label, oracle: _Oracle, candidates: list[str], max_words: int):
    if not candidates:
        return text, 0, False
    current, added = text, 0
    cur_p = oracle.p_true([current], label)[0]
    for _ in range(max_words):
        texts = [f"{current} {w}" for w in candidates]
        p = oracle.p_true(texts, label)
        j = int(np.argmin(p))
        if p[j] >= cur_p:
            break
        current, cur_p, added = texts[j], float(p[j]), added + 1
        if oracle.label(current) != label:
            return current, added, True
    return current, added, False


def build_candidates(texts: list[str], labels: np.ndarray, per_label: int) -> dict:
    """For each class, the words that appear most often in OTHER classes relative to this one."""
    classes = list(dict.fromkeys(labels.tolist()))
    doc_freq = {c: Counter() for c in classes}
    sizes = Counter(labels.tolist())
    for t, y in zip(texts, labels.tolist()):
        doc_freq[y].update(set(_WORD.findall(t.lower())))
    out = {}
    for c in classes:
        scores: dict[str, float] = {}
        vocab = set().union(*[set(doc_freq[o]) for o in classes if o != c]) if len(classes) > 1 else set()
        for w in vocab:
            other = sum(doc_freq[o][w] for o in classes if o != c) / max(1, sum(sizes[o] for o in classes if o != c))
            own = doc_freq[c][w] / max(1, sizes[c])
            scores[w] = other - own
        out[c] = [w for w, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:per_label]]
    return out


def run_attack(name: str, params: dict, clf, texts: list[str], labels: np.ndarray, n_examples: int = 8) -> dict:
    key = name.lower()
    unknown = set(params) - ALLOWED_PARAMS[key]
    if unknown:
        raise ValueError(f"{name} does not accept parameter(s): {sorted(unknown)}. "
                         f"Allowed: {sorted(ALLOWED_PARAMS[key])}")
    p = {**DEFAULTS[key], **params}
    n = len(texts)
    started = time.perf_counter()

    clean_pred = clf.predict(texts)
    correct = clean_pred == labels
    candidates = build_candidates(texts, labels, p["candidates"]) if key == "word_insertion" else {}

    adv_texts = list(texts)
    words_changed, queries = [], []
    examples: list[tuple] = []
    for i in np.flatnonzero(correct):
        oracle = _Oracle(clf)
        label = labels[i]
        if key == "char_perturb":
            adv, changed, _ = _attack_char(texts[i], label, oracle, p["max_words"])
        else:
            adv, changed, _ = _attack_insert(texts[i], label, oracle, candidates.get(label, []), p["max_words"])
        adv_texts[i] = adv
        words_changed.append(changed)
        queries.append(oracle.queries)

    adv_pred = clf.predict(adv_texts)
    flipped = correct & (adv_pred != labels)
    for i in np.flatnonzero(flipped)[:n_examples]:
        examples.append((texts[i], adv_texts[i], labels[i].item() if hasattr(labels[i], "item") else labels[i],
                         adv_pred[i].item() if hasattr(adv_pred[i], "item") else adv_pred[i]))

    n_correct = int(correct.sum())
    return {
        "attack": name.upper(),
        "params": params,
        "n_samples": n,
        "clean_accuracy": n_correct / n,
        "adversarial_accuracy": float((adv_pred == labels).mean()),
        "attack_success_rate": (int(flipped.sum()) / n_correct) if n_correct else None,
        "mean_words_changed": float(np.mean(words_changed)) if words_changed else 0.0,
        "mean_queries": float(np.mean(queries)) if queries else 0.0,
        "seconds": round(time.perf_counter() - started, 2),
        "examples": examples,
        "error": None,
    }

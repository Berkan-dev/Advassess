"""Attacks that run when the config lists none.

The suites are meant to be a fair first look, not a full audit: a few strengths of the
standard attacks plus slower ones capped with `samples`. Settings are tuned so each attack
actually does something on typical models (see the README note on weak settings).
"""
from __future__ import annotations


def _r(value: float) -> float:
    return float(f"{value:.4g}")


def art_suite(lo: float, hi: float, n_features: int) -> list[dict]:
    """Attack sizes are fractions of the valid input range (hi - lo)."""
    rng = hi - lo
    e = lambda f: _r(f * rng)
    gamma = 0.1 if n_features >= 100 else 0.4
    return [
        *({"name": "fgsm", "params": {"eps": e(f)}} for f in (0.05, 0.1, 0.2, 0.3)),
        {"name": "pgd", "params": {"eps": e(0.1), "eps_step": e(0.01), "max_iter": 40}},
        {"name": "pgd", "params": {"eps": e(0.3), "eps_step": e(0.03), "max_iter": 40}},
        {"name": "deepfool", "params": {"max_iter": 50}, "samples": 200},
        {"name": "jsma", "params": {"theta": _r(rng), "gamma": gamma}, "samples": 100},
        {"name": "cw", "params": {"max_iter": 30, "binary_search_steps": 5, "learning_rate": e(0.05),
                                  "initial_const": 1.0}, "samples": 50},
        {"name": "ead", "params": {"max_iter": 30, "binary_search_steps": 5, "learning_rate": e(0.05),
                                   "initial_const": 1.0}, "samples": 50},
        {"name": "square", "params": {"eps": e(0.3), "max_iter": 200, "nb_restarts": 1}, "samples": 100},
        {"name": "hsja", "params": {"max_iter": 10, "max_eval": 500, "init_eval": 50, "init_size": 100},
         "samples": 30},
        {"name": "autoattack", "params": {"eps": e(0.3), "eps_step": e(0.03)}, "samples": 100},
    ]


TEXT_SUITE = [
    {"name": "char_perturb", "params": {"max_words": 3}},
    {"name": "word_insertion", "params": {"max_words": 5, "candidates": 50}},
]

# A modest first scan across the main LLM weakness families; each probe is capped by
# run.max_prompts_per_probe and run.generations. Probes whose detectors download a classifier
# (toxicity and similar) are left out so the default works offline and without a large download.
LLM_SUITE = [
    {"name": name, "params": {}} for name in (
        "promptinject.HijackHateHumans", "promptinject.HijackKillHumans",
        "dan.Dan_11_0", "dan.DanInTheWild", "encoding.InjectBase64",
        "latentinjection.LatentInjectionResume", "goodside.Tag", "goodside.WhoIsRiley",
    )
]


def blackbox_extra(lo: float, hi: float) -> list[dict]:
    """Added when the model has no gradients (trees, forests, pipelines): only query-based attacks apply."""
    return [
        {"name": "boundary", "params": {"max_iter": 150}, "samples": 20},
        {"name": "signopt", "params": {"max_iter": 60, "query_limit": 1000}, "samples": 20},
        {"name": "zoo", "params": {"max_iter": 100, "binary_search_steps": 3, "nb_parallel": 32,
                                   "initial_const": 1.0, "learning_rate": _r(0.1 * (hi - lo))}, "samples": 10},
    ]

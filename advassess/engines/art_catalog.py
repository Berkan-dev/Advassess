"""Which ART evasion attacks the tool can run, and what to call them.

This file holds names only (no ART import), so config checks stay fast and work
even when ART is not installed. Attack classes are imported when a run starts.

category
    white-box   needs the model's gradients (the model file itself)
    black-box   only asks the model for predictions
    mixed       combines both kinds (AutoAttack)

VERIFIED lists attacks that were run end to end during development; the rest are available but untested.
"""
from __future__ import annotations

# ART class name -> (category, one-line description)
CATALOG: dict[str, tuple[str, str]] = {
    # --- gradient-based, L-inf / L2 budgets
    "FastGradientMethod": ("white-box", "One-step gradient attack (FGSM)"),
    "BasicIterativeMethod": ("white-box", "FGSM repeated in small steps (BIM)"),
    "ProjectedGradientDescent": ("white-box", "Iterated gradient steps with projection (PGD)"),
    "MomentumIterativeMethod": ("white-box", "Iterative attack with momentum (MIM)"),
    "AutoProjectedGradientDescent": ("white-box", "PGD with automatic step-size schedule (APGD)"),
    "AutoConjugateGradient": ("white-box", "Conjugate-gradient attack (ACG)"),
    "RescalingAutoConjugateGradient": ("white-box", "ACG with rescaling (RACG)"),
    "VirtualAdversarialMethod": ("white-box", "Virtual adversarial perturbation (VAT)"),
    "ShadowAttack": ("white-box", "Visible but natural-looking 'shadow' perturbation"),
    "CompositeAdversarialAttackPyTorch": ("white-box", "Semantic attack: hue, brightness, rotation and more"),
    # --- minimal-perturbation attacks
    "CarliniL2Method": ("white-box", "Carlini & Wagner, L2 norm"),
    "CarliniLInfMethod": ("white-box", "Carlini & Wagner, L-inf norm"),
    "CarliniL0Method": ("white-box", "Carlini & Wagner, L0 norm (fewest changed pixels)"),
    "DeepFool": ("white-box", "Smallest step to the nearest decision boundary"),
    "NewtonFool": ("white-box", "Newton-style step against the top class probability"),
    "ElasticNet": ("white-box", "Elastic-net attack (EAD): sparse and small changes"),
    "SaliencyMapMethod": ("white-box", "Jacobian-based saliency map attack (JSMA)"),
    "Wasserstein": ("white-box", "Attack under a Wasserstein-distance budget"),
    # --- query-only attacks
    "BoundaryAttack": ("black-box", "Decision-based random walk along the boundary"),
    "HopSkipJump": ("black-box", "Decision-based attack (HSJA)"),
    "SignOPTAttack": ("black-box", "Decision-based sign-OPT attack"),
    "SquareAttack": ("black-box", "Score-based random-square attack"),
    "ZooAttack": ("black-box", "Zeroth-order optimization attack (ZOO)"),
    "SimBA": ("black-box", "Simple black-box attack"),
    "PixelAttack": ("black-box", "Few-pixel attack (differential evolution)"),
    "ThresholdAttack": ("black-box", "Threshold attack (differential evolution)"),
    "SpatialTransformation": ("black-box", "Rotations and translations"),
    "AutoAttack": ("mixed", "Ensemble of strong attacks, a robustness benchmark"),
}

# short name (lower-case) -> ART class name. Lower-case class names also work.
ALIASES: dict[str, str] = {
    "fgsm": "FastGradientMethod", "fgm": "FastGradientMethod",
    "bim": "BasicIterativeMethod", "pgd": "ProjectedGradientDescent",
    "mim": "MomentumIterativeMethod", "apgd": "AutoProjectedGradientDescent",
    "acg": "AutoConjugateGradient", "racg": "RescalingAutoConjugateGradient",
    "vat": "VirtualAdversarialMethod", "shadow": "ShadowAttack",
    "composite": "CompositeAdversarialAttackPyTorch",
    "cw": "CarliniL2Method", "cw_l2": "CarliniL2Method", "cw_linf": "CarliniLInfMethod",
    "cw_l0": "CarliniL0Method", "deepfool": "DeepFool", "newtonfool": "NewtonFool",
    "ead": "ElasticNet", "elasticnet": "ElasticNet", "jsma": "SaliencyMapMethod",
    "wasserstein": "Wasserstein", "boundary": "BoundaryAttack", "hsja": "HopSkipJump",
    "hopskipjump": "HopSkipJump", "signopt": "SignOPTAttack",
    "square": "SquareAttack", "zoo": "ZooAttack", "simba": "SimBA", "pixel": "PixelAttack",
    "threshold": "ThresholdAttack", "spatial": "SpatialTransformation",
    "autoattack": "AutoAttack",
}

# ART evasion attacks this tool does NOT run, with the reason.
UNSUPPORTED: dict[str, str] = {
    "UniversalPerturbation": "builds one perturbation from a whole dataset, not per input",
    "TargetedUniversalPerturbation": "builds one perturbation from a whole dataset, not per input",
    "AdversarialPatch": "trains a patch on a dataset, not per input",
    "AdversarialPatchNumpy": "trains a patch on a dataset, not per input",
    "AdversarialPatchPyTorch": "trains a patch on a dataset, not per input",
    "AdversarialPatchTensorFlowV2": "trains a patch on a dataset, not per input",
    "DPatch": "needs an object-detection model",
    "RobustDPatch": "needs an object-detection model",
    "OverloadPyTorch": "needs an object-detection model",
    "AdversarialTexturePyTorch": "needs an object-tracking model",
    "OverTheAirFlickeringPyTorch": "needs a video model",
    "FrameSaliencyAttack": "needs a video model",
    "CarliniWagnerASR": "needs a speech-recognition model",
    "ImperceptibleASR": "needs a speech-recognition model",
    "ImperceptibleASRPyTorch": "needs a speech-recognition model",
    "MalwareGDTensorFlow": "needs a TensorFlow malware classifier",
    "DecisionTreeAttack": "needs a scikit-learn decision tree",
    "HighConfidenceLowUncertainty": "needs a Gaussian-process classifier",
    "FeatureAdversariesNumpy": "needs a target example and feature layer",
    "FeatureAdversariesPyTorch": "needs a target example and feature layer",
    "FeatureAdversariesTensorFlowV2": "needs a target example and feature layer",
    "LowProFool": "needs per-feature importance weights and bounds (tabular only)",
    "GRAPHITEBlackbox": "needs a template and patch-region setup",
    "GRAPHITEWhiteboxPyTorch": "needs a template and patch-region setup",
    "LaserAttack": "needs a laser-pattern setup",
    "SNAL": "needs an object-detection model",
    "GeoDA": "ART's implementation uses np.mat, which NumPy 2 removed (fails on current NumPy)",
    "ProjectedGradientDescentNumpy": "use PGD, which picks the right implementation",
    "ProjectedGradientDescentPyTorch": "use PGD, which picks the right implementation",
    "ProjectedGradientDescentTensorFlowV2": "use PGD, which picks the right implementation",
}

VERIFIED: frozenset[str] = frozenset({
    "FastGradientMethod",
    "BasicIterativeMethod",
    "ProjectedGradientDescent",
    "MomentumIterativeMethod",
    "AutoProjectedGradientDescent",
    "AutoConjugateGradient",
    "RescalingAutoConjugateGradient",
    "VirtualAdversarialMethod",
    "CarliniL2Method",
    "CarliniLInfMethod",
    "CarliniL0Method",
    "DeepFool",
    "NewtonFool",
    "ElasticNet",
    "SaliencyMapMethod",
    "Wasserstein",
    "BoundaryAttack",
    "HopSkipJump",
    "SignOPTAttack",
    "SquareAttack",
    "ZooAttack",
    "SimBA",
    "PixelAttack",
    "ThresholdAttack",
    "SpatialTransformation",
    "AutoAttack",
})


def resolve_name(name: str) -> str | None:
    """Accept an alias or an ART class name in any letter case; return the ART class name."""
    key = name.strip().lower()
    if key in ALIASES:
        return ALIASES[key]
    for cls in CATALOG:
        if cls.lower() == key:
            return cls
    return None


def known_names() -> set[str]:
    return set(ALIASES) | {c.lower() for c in CATALOG}

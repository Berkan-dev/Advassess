"""Which ART evasion attacks the tool can run, and what to call them.

This file holds names only (no ART import), so config checks stay fast and work
even when ART is not installed. Attack classes are imported when a run starts.

category
    white-box   needs the model's gradients (the model file itself)
    black-box   only asks the model for predictions
    mixed       combines both kinds (AutoAttack)

VERIFIED lists attacks that were run end to end during development; the rest are available but untested.

scope (SCOPE below)
    input     attacks each test input on its own (the default)
    dataset   learns ONE perturbation or patch from a fit split, then is scored on held-out inputs
    setup     needs extra setup (a guide example, feature weights, template ...) that the tool derives or the config gives

task (TASK below) says which kind of model an attack needs. Anything not listed is a "classification" attack.
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
    # --- dataset-level: learn one perturbation / patch on a fit split, score it on held-out inputs
    "UniversalPerturbation": ("white-box", "One perturbation that fools many inputs (uses an inner attack such as DeepFool)"),
    "TargetedUniversalPerturbation": ("white-box", "One perturbation that pushes many inputs to a chosen class"),
    "AdversarialPatch": ("white-box", "Patch learned on a dataset, pasted onto inputs (implementation picked automatically)"),
    "AdversarialPatchNumpy": ("white-box", "Adversarial patch, NumPy implementation"),
    "AdversarialPatchPyTorch": ("white-box", "Adversarial patch, PyTorch implementation"),
    "AdversarialPatchTensorFlowV2": ("white-box", "Adversarial patch, TensorFlow implementation"),
    # --- setup-heavy (the tool derives sensible defaults; override under params)
    "FeatureAdversariesNumpy": ("white-box", "Match another input's internal features (feature adversaries)"),
    "FeatureAdversariesPyTorch": ("white-box", "Feature adversaries, PyTorch implementation"),
    "FeatureAdversariesTensorFlowV2": ("white-box", "Feature adversaries, TensorFlow implementation"),
    "LowProFool": ("white-box", "Tabular attack that avoids changing important features"),
    "GRAPHITEBlackbox": ("black-box", "Robust physical-style patch from a heatmap (query-only)"),
    "GRAPHITEWhiteboxPyTorch": ("white-box", "Robust physical-style patch from gradients"),
    "LaserBeamAttack": ("black-box", "Laser-beam light artefacts searched by queries"),
    # --- other model types (see TASK)
    "DecisionTreeAttack": ("white-box", "Walks a scikit-learn decision tree to the nearest other leaf"),
    "HighConfidenceLowUncertainty": ("white-box", "Gaussian-process attack: confident wrong answer with low uncertainty"),
    "MalwareGDTensorFlow": ("white-box", "Appends bytes to a binary to evade a TensorFlow malware classifier"),
    "OverTheAirFlickeringPyTorch": ("white-box", "Flickering perturbation of a video clip"),
    "FrameSaliencyAttack": ("white-box", "Perturbs the most salient video frames using an inner attack"),
    "DPatch": ("white-box", "Patch that hides or misleads an object detector"),
    "RobustDPatch": ("white-box", "Detector patch robust to crops, brightness and rotation"),
    "OverloadPyTorch": ("white-box", "Floods an object detector with candidate boxes to slow it down"),
    "SNAL": ("white-box", "Slows an object detector by adding candidate boxes (latency attack)"),
    "AdversarialTexturePyTorch": ("white-box", "Texture patch that derails a visual object tracker"),
    "CarliniWagnerASR": ("white-box", "Targeted audio perturbation that makes a speech model transcribe chosen text"),
    "ImperceptibleASR": ("white-box", "Targeted audio perturbation kept under the hearing threshold"),
    "ImperceptibleASRPyTorch": ("white-box", "Imperceptible ASR attack, PyTorch implementation"),
}

SCOPE: dict[str, str] = {
    "UniversalPerturbation": "dataset", "TargetedUniversalPerturbation": "dataset",
    "AdversarialPatch": "dataset", "AdversarialPatchNumpy": "dataset",
    "AdversarialPatchPyTorch": "dataset", "AdversarialPatchTensorFlowV2": "dataset",
    "FeatureAdversariesNumpy": "setup", "FeatureAdversariesPyTorch": "setup", "FeatureAdversariesTensorFlowV2": "setup",
    "LowProFool": "setup", "GRAPHITEBlackbox": "setup", "GRAPHITEWhiteboxPyTorch": "setup", "LaserBeamAttack": "setup",
}

# Attacks that need something other than an ordinary classifier. task -> attack class names.
TASKS: dict[str, tuple[str, ...]] = {
    "video": ("OverTheAirFlickeringPyTorch", "FrameSaliencyAttack"),
    "object_detection": ("DPatch", "RobustDPatch", "OverloadPyTorch", "SNAL"),
    "object_tracking": ("AdversarialTexturePyTorch",),
    "speech": ("CarliniWagnerASR", "ImperceptibleASR", "ImperceptibleASRPyTorch"),
    "malware": ("MalwareGDTensorFlow",),
    "gaussian_process": ("HighConfidenceLowUncertainty",),
}
TASK: dict[str, str] = {cls: task for task, names in TASKS.items() for cls in names}
# Classification attacks that only make sense for one kind of classifier (checked with a clear message).
NEEDS_TABULAR = {"LowProFool"}
NEEDS_SKLEARN_TREE = {"DecisionTreeAttack"}
ALL_TASKS = ("classification", *TASKS)

# ART classes that its evasion package does not export at the top level: class -> module that defines it.
SPECIAL_MODULES = {"LaserBeamAttack": "art.attacks.evasion.laser_attack.laser_attack"}


def task_of(class_name: str) -> str:
    return TASK.get(class_name, "classification")


def scope_of(class_name: str) -> str:
    return SCOPE.get(class_name, "input")

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
    "universal": "UniversalPerturbation", "targeted_universal": "TargetedUniversalPerturbation",
    "patch": "AdversarialPatch", "featureadv": "FeatureAdversariesPyTorch", "lowprofool": "LowProFool",
    "graphite": "GRAPHITEBlackbox", "graphite_wb": "GRAPHITEWhiteboxPyTorch", "laser": "LaserBeamAttack",
    "dtattack": "DecisionTreeAttack", "hclu": "HighConfidenceLowUncertainty", "malwaregd": "MalwareGDTensorFlow",
    "flicker": "OverTheAirFlickeringPyTorch", "framesaliency": "FrameSaliencyAttack",
    "dpatch": "DPatch", "rdpatch": "RobustDPatch", "overload": "OverloadPyTorch",
    "texture": "AdversarialTexturePyTorch", "cw_asr": "CarliniWagnerASR",
    "imperceptible_asr": "ImperceptibleASR",
}

# ART evasion attacks this tool does NOT run, with the reason.
UNSUPPORTED: dict[str, str] = {
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

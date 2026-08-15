"""
pipeline_utils.py

Single import point for every script in the benchmark:
  - synthesis training / inference
  - segmentation training / inference (PASSION, mmFormer, RFNet, ...)
  - Oracle evaluation

Rule of thumb: if a script needs a number (rotation range, patch size,
noise sigma, split ratio, ...) or a stochastic decision (augment or not,
which patients are in test), it must come from here — never hardcoded
or re-derived locally. This is what makes "identical policy across
models" an enforced property instead of a documented intention.
"""

import json
import os
import random
from pathlib import Path
from typing import Literal, Optional, Dict, Any, List, Union

import numpy as np
import yaml

# Resolve default config path searching root and configs/
_HERE = Path(__file__).parent
_ROOT = _HERE.parent.parent
if (_ROOT / "config.yaml").exists():
    CONFIG_PATH = _ROOT / "config.yaml"
elif (_ROOT / "configs" / "default.yaml").exists():
    CONFIG_PATH = _ROOT / "configs" / "default.yaml"
elif (_HERE / "config.yaml").exists():
    CONFIG_PATH = _HERE / "config.yaml"
else:
    CONFIG_PATH = _ROOT / "config.yaml"


class Config(dict):
    """Dict subclass that allows both dict and attribute-style access: cfg.training.batch_size"""
    def __getattr__(self, key):
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, Config):
                return Config(val)
            return val
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _set_nnunet_env_vars(cfg: Config, project_root: Path) -> None:
    """Sets nnUNet mandatory environment variables from config.

    Paths are resolved relative to project_root (the directory containing
    the configs/ folder) so they are correct regardless of the CWD from
    which the training script is launched.
    """
    if "nnunet" not in cfg:
        return
    nnunet_cfg = cfg["nnunet"]
    mapping = {
        "nnUNet_raw": nnunet_cfg.get("raw", "data/raw"),
        "nnUNet_preprocessed": nnunet_cfg.get("preprocessed", "data/processed"),
        "nnUNet_results": nnunet_cfg.get("results", "checkpoints"),
    }
    for env_var, rel_path in mapping.items():
        if env_var not in os.environ:
            os.environ[env_var] = str((project_root / rel_path).resolve())


def load_config(path: Path = CONFIG_PATH, model_name: Optional[str] = None) -> Config:
    """Load the single shared config. Call this at the top of every script."""
    path = Path(path)
    if not path.exists():
        alt = _ROOT / "configs" / "default.yaml"
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Config file not found at {path}")

    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    if path.name == "default.yaml" and path.parent.name == "configs":
        project_root = path.parent.parent
        configs_dir = path.parent
    else:
        project_root = path.parent
        configs_dir = project_root / "configs"

    if model_name is not None:
        model_config_path = configs_dir / "models" / f"{model_name}.yaml"
        if model_config_path.exists():
            with open(model_config_path, "r") as f:
                model_cfg = yaml.safe_load(f)
            cfg_dict = _deep_merge(cfg_dict, model_cfg)
        else:
            print(f"[Config] Warning: No model config found at {model_config_path}")

    config = Config(cfg_dict)
    _set_nnunet_env_vars(config, project_root=project_root)
    return config


# ---------------------------------------------------------------------------
# Global reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch. Call once, at the very top of every script,
    using cfg['seed'] — never a locally chosen seed."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def worker_init_fn(worker_id: int, base_seed: int):
    """Pass to DataLoader(worker_init_fn=partial(worker_init_fn, base_seed=cfg['seed']))
    so multi-worker DataLoaders don't silently duplicate augmentation streams."""
    worker_seed = (base_seed + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Canonical patient-wise split — generated ONCE, loaded everywhere
# ---------------------------------------------------------------------------

def make_splits(patient_ids: List[str], cfg: dict, out_path: Optional[Path] = None) -> Dict[str, Any]:
    """Run exactly once (e.g. via a standalone make_splits.py), NOT inside
    individual training scripts. Writes {train, val, test} patient ID lists
    to cfg['paths']['splits_file']. Every other script only ever READS
    this file via load_splits().
    """
    rng = random.Random(cfg["seed"])
    ids = sorted(patient_ids)  # sort first so shuffle is deterministic given seed
    rng.shuffle(ids)

    n = len(ids)
    ratios = cfg["split"]["ratios"]
    n_train = int(round(n * ratios["train"]))
    n_val = int(round(n * ratios["val"]))

    splits = {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
        "seed": cfg["seed"],
        "ratios": ratios,
    }

    # sanity: no overlap, full coverage
    assert set(splits["train"]) & set(splits["val"]) == set(), "Overlap between train and val"
    assert set(splits["train"]) & set(splits["test"]) == set(), "Overlap between train and test"
    assert set(splits["val"]) & set(splits["test"]) == set(), "Overlap between val and test"
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == n, "Split does not cover all patients"

    out_path = out_path or Path(cfg["paths"]["splits_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    return splits


def load_splits(cfg: dict) -> Dict[str, Any]:
    """Every training/eval script (synthesis, segmentation, Oracle) calls
    this — never make_splits() directly — so all stages share identical
    patient assignment and leakage across stages/models is structurally
    impossible."""
    path = Path(cfg["paths"]["splits_file"])
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run make_splits.py once before any "
            f"training/eval script. Do not generate splits inside a "
            f"training script."
        )
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Normalization — identical function used for real inputs AND
# for re-normalizing synthesis outputs before they reach segmentation
# ---------------------------------------------------------------------------

def zscore_normalize(volume: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-modality, per-patient z-score. mask (if given) restricts the
    mean/std computation to foreground voxels (recommended, matches
    preprocessing.foreground_crop convention in config.yaml)."""
    region = volume[mask > 0] if mask is not None else volume
    if len(region) == 0 or region.std() == 0:
        return volume.astype(np.float32)
    mean = region.mean()
    std = region.std() + 1e-8
    normalized = (volume - mean) / std
    if mask is not None:
        normalized[mask == 0] = 0.0
    return normalized.astype(np.float32)


def renormalize_synthetic_output(volume: np.ndarray, cfg: dict, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Junction-point function: converts a generator's raw output
    (e.g. tanh [-1,1]) into the SAME z-score space real modalities are
    in, using cfg['junction'] so this never has to be reimplemented
    per-generator. Every synthesis model's output MUST pass through
    this before being handed to a segmentation model."""
    src_range = cfg["junction"]["synthesis_output_range"]
    if src_range == "tanh_[-1,1]":
        volume = (volume + 1.0) / 2.0  # -> [0, 1]
    elif src_range == "sigmoid_[0,1]":
        pass  # already in [0, 1], proceed to z-score
    elif src_range == "raw":
        pass  # assume raw intensity, proceed to z-score
    else:
        raise ValueError(
            f"Unknown synthesis_output_range: '{src_range}'. "
            f"Expected one of: 'tanh_[-1,1]', 'sigmoid_[0,1]', 'raw'."
        )
    # then treat as a normal intensity volume and z-score it
    return zscore_normalize(volume, mask=mask)


# ---------------------------------------------------------------------------
# Patch sampling — shared by synthesis and segmentation, train and eval
# ---------------------------------------------------------------------------

def get_patch_size(cfg: dict) -> tuple[int, int, int]:
    return tuple(cfg["patch"]["size"])


def sample_patch_coords(volume_shape, patch_size, mode: Literal["random", "center"], rng: random.Random) -> tuple:
    """mode='random' for training (both synthesis and segmentation),
    mode='center' for ALL val/test evaluation (both stages) — deterministic,
    no exceptions. Passing the same `rng` (seeded from cfg['seed']) across
    scripts keeps behavior reproducible run-to-run."""
    starts = []
    for dim_size, p_size in zip(volume_shape, patch_size):
        if mode == "center":
            start = max(0, (dim_size - p_size) // 2)
        elif mode == "random":
            max_start = max(0, dim_size - p_size)
            start = rng.randint(0, max_start) if max_start > 0 else 0
        else:
            raise ValueError(f"unknown patch sampling mode: {mode}")
        starts.append(start)
    return tuple(starts)


# ---------------------------------------------------------------------------
# Augmentation policies — thin wrappers that read directly from config.yaml
# ---------------------------------------------------------------------------

def get_synthesis_augmentation_params(cfg: dict, split: Literal["train", "val", "test"]) -> Optional[dict]:
    """Returns the augmentation param dict for synthesis models, or None
    if split != 'train' (i.e. val/test are ALWAYS deterministic — assert
    this return value drives augment=False downstream)."""
    if split != "train":
        return None
    return cfg.get("augmentation_synthesis")


def get_segmentation_augmentation_params(cfg: dict, split: Literal["train", "val", "test"]) -> Optional[dict]:
    """Same contract as above, for segmentation models (PASSION/mmFormer/
    RFNet). Deliberately a SEPARATE function from the synthesis one even
    though the logic is identical, so it's obvious at the call site which
    policy a script is using — reduces risk of copy-paste cross-wiring."""
    if split != "train":
        return None
    return cfg.get("augmentation_segmentation")


# ---------------------------------------------------------------------------
# Missing-modality simulation — applied AFTER augmentation + normalization
# ---------------------------------------------------------------------------

def apply_missing_modality(volume_dict: dict, scenario_name: str, cfg: dict) -> dict:
    """volume_dict: {"t1": arr, "t1ce": arr, "t2": arr, "flair": arr}
    (already augmented + normalized). Zeros out / masks the channels
    specified by the named scenario in config.yaml. Call this LAST in
    the per-sample pipeline, per pipeline ordering note in config.yaml."""
    scenarios = {
        s["name"].upper(): [d.lower() for d in s["drop"]]
        for s in cfg["missing_modality"]["scenarios"]
    }
    for s in cfg["missing_modality"]["scenarios"]:
        scenarios[s["name"]] = [d.lower() for d in s["drop"]]

    if scenario_name not in scenarios and scenario_name.upper() not in scenarios:
        raise ValueError(f"unknown missing-modality scenario: {scenario_name}. Available: {list(scenarios.keys())}")

    drop_list = scenarios.get(scenario_name, scenarios.get(scenario_name.upper()))

    out = {}
    for mod, arr in volume_dict.items():
        out[mod] = np.zeros_like(arr) if mod.lower() in drop_list else arr
    return out


def get_modality_order(cfg: dict) -> list[str]:
    """Canonical channel order — use this everywhere a volume_dict is
    stacked into a model input tensor, never a locally hardcoded list."""
    return cfg["modalities"]["order"]


def stack_modalities(volume_dict: dict, cfg: dict) -> np.ndarray:
    order = get_modality_order(cfg)
    return np.stack([volume_dict[m] if m in volume_dict else volume_dict[m.lower()] for m in order], axis=0)


# ---------------------------------------------------------------------------
# Evaluation — one metric implementation, pinned, used everywhere
# ---------------------------------------------------------------------------

def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, label_ids: list[int]) -> float:
    pred_bin = np.isin(pred_mask, label_ids)
    gt_bin = np.isin(gt_mask, label_ids)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    denom = pred_bin.sum() + gt_bin.sum()
    if denom == 0:
        return 1.0  # both empty -> perfect agreement by convention; document this choice
    return float(2.0 * intersection / denom)


def get_label_groups(cfg: dict) -> dict:
    """Returns {'whole_tumor': [...], 'tumor_core': [...], 'enhancing_tumor': [...]}
    Use these ids for EVERY Dice/HD95 call across the whole project —
    never redefine the BraTS label groupings locally in an analysis script."""
    return cfg["labels"]


# ---------------------------------------------------------------------------
# Sanity-check helper — run at the start of any script as a guard rail
# ---------------------------------------------------------------------------

def assert_pipeline_consistency(cfg: dict, check_splits_exist: bool = True) -> None:
    """Cheap structural checks that catch the most common drift bugs.
    Call this once after load_config() in every script."""
    if check_splits_exist:
        splits_file = Path(cfg["paths"]["splits_file"])
        assert splits_file.exists(), (
            f"splits_file ({splits_file}) missing — run make_splits.py before any train/eval script"
        )
    patch_size = cfg["patch"]["size"]
    assert isinstance(patch_size, list) and len(patch_size) == 3 and all(s > 0 for s in patch_size), (
        f"patch size must be a list of 3 positive integers, got {patch_size}"
    )
    assert cfg["missing_modality"]["apply_order"] == "after_augmentation_after_normalization", (
        "missing-modality masking must be applied after augmentation and "
        "normalization — check config.yaml if this assertion fires"
    )
    norm_a = cfg["preprocessing"]["normalization"]
    norm_b = cfg["junction"]["renormalize_to"]
    assert norm_a == norm_b, (
        f"preprocessing normalization ({norm_a}) and junction renormalization "
        f"target ({norm_b}) must match, or synthesis outputs will be fed to "
        f"segmentation models in a different intensity space than real inputs"
    )

import copy
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


class Config(dict):
    """Dict subclass that allows attribute-style access: cfg.training.batch_size"""
    def __getattr__(self, key):
        try:
            val = self[key]
            if isinstance(val, dict):
                return Config(val)
            return val
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


def load_config(
    model_name: Optional[str] = None,
    config_dir: str = "configs"
) -> Config:
    """
    Loads the base default.yaml config and optionally merges a
    model-specific override config from configs/models/<model_name>.yaml.

    Also sets mandatory nnUNet environment variables from config.

    Args:
        model_name: Optional model name (e.g. 'nnunet', 'pix2pix').
        config_dir: Directory containing config files.

    Returns:
        Merged Config object.
    """
    config_dir = Path(config_dir)
    default_path = config_dir / "default.yaml"

    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found at {default_path}")

    with open(default_path, "r") as f:
        cfg = yaml.safe_load(f)

    if model_name is not None:
        model_config_path = config_dir / "models" / f"{model_name}.yaml"
        if model_config_path.exists():
            with open(model_config_path, "r") as f:
                model_cfg = yaml.safe_load(f)
            cfg = _deep_merge(cfg, model_cfg)
        else:
            print(f"[Config] Warning: No model config found at {model_config_path}")

    config = Config(cfg)
    _set_nnunet_env_vars(config, project_root=config_dir.parent)
    return config


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

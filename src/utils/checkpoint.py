import torch
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


class EarlyStopping:
    """
    Early stopping tracker. Stops training when a monitored metric
    does not improve for `patience` consecutive validation checks.
    """

    def __init__(self, patience: int = 20, mode: str = "max", min_delta: float = 1e-4):
        """
        Args:
            patience: Number of epochs without improvement before stopping.
            mode: 'max' for metrics like Dice/PSNR, 'min' for loss.
            min_delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, score: float) -> bool:
        """
        Call after each validation epoch.

        Args:
            score: Current validation metric value.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        self.best_score = None
        self.counter = 0
        self.should_stop = False


class CheckpointManager:
    """
    Manages saving and loading of model checkpoints.
    Always saves the best checkpoint based on monitored metric.
    """

    def __init__(self, checkpoint_dir: str, model_name: str, scenario: str):
        """
        Args:
            checkpoint_dir: Base directory for checkpoints (e.g., 'checkpoints').
            model_name: Model identifier (e.g., 'pix2pix', 'nnunet').
            scenario: Scenario identifier (e.g., 'S1').
        """
        self.save_dir = Path(checkpoint_dir) / model_name / scenario
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.save_dir / "best.pth"
        self.last_path = self.save_dir / "last.pth"

    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             epoch: int, metric: float, extra: Optional[Dict[str, Any]] = None,
             is_best: bool = False) -> None:
        """
        Saves a checkpoint.

        Args:
            model: The PyTorch model to save.
            optimizer: The optimizer state to save.
            epoch: Current training epoch.
            metric: Current validation metric score.
            extra: Optional dict of additional metadata to include.
            is_best: If True, also saves as best.pth.
        """
        state = {
            "epoch": epoch,
            "metric": metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        if extra:
            state.update(extra)

        torch.save(state, self.last_path)

        if is_best:
            shutil.copyfile(self.last_path, self.best_path)
            print(f"[Checkpoint] Best checkpoint saved → {self.best_path} (metric={metric:.4f})")

    def load(self, model: torch.nn.Module, path: Optional[str] = None,
             optimizer: Optional[torch.optim.Optimizer] = None,
             strict: bool = True, device: str = "cpu") -> Dict[str, Any]:
        """
        Loads a checkpoint into a model.

        Args:
            model: Model to load weights into.
            path: Path to checkpoint file. Defaults to best.pth.
            optimizer: Optional optimizer to restore state into.
            strict: Whether to enforce strict key matching.
            device: Device to map tensors to.

        Returns:
            The full checkpoint dict (epoch, metric, etc.).
        """
        load_path = Path(path) if path else self.best_path
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint = torch.load(load_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]

        # Strip 'module.' prefix if weights were saved from DDP and model is non-DDP
        if not hasattr(model, "module") and any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=strict)

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        metric_val = checkpoint.get('metric')
        metric_str = f"{metric_val:.4f}" if metric_val is not None else "N/A"
        print(f"[Checkpoint] Loaded from {load_path} (epoch={checkpoint.get('epoch')}, metric={metric_str})")
        return checkpoint

    def best_exists(self) -> bool:
        return self.best_path.exists()

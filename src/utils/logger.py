import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


class ExperimentLogger:
    """
    Logs per-epoch training metrics to a CSV file and optionally to TensorBoard.

    CSV schema: epoch, model, scenario, phase, loss, dice_wt, dice_tc, dice_et,
                dice_mean, hd95_mean, psnr, ssim, mae, mse, timestamp
    """

    COLUMNS = [
        "epoch", "model", "scenario", "phase",
        "loss",
        "dice_wt", "dice_tc", "dice_et", "dice_mean",
        "hd95_wt", "hd95_tc", "hd95_et", "hd95_mean",
        "psnr", "ssim", "mae", "mse",
        "timestamp"
    ]

    def __init__(
        self,
        log_dir: str,
        model_name: str,
        scenario: str,
        use_tensorboard: bool = True
    ):
        """
        Args:
            log_dir: Base directory for logs (e.g., 'results/logs').
            model_name: Model identifier (e.g., 'pix2pix').
            scenario: Scenario identifier (e.g., 'S1').
            use_tensorboard: Whether to write TensorBoard events.
        """
        self.model_name = model_name
        self.scenario = scenario

        run_dir = Path(log_dir) / model_name / scenario
        run_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = run_dir / "metrics.csv"
        self._init_csv()

        self.writer = None
        if use_tensorboard and HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    def _init_csv(self):
        if os.environ.get("LOCAL_RANK", "0") != "0":
            return
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writeheader()

    def log(
        self,
        epoch: int,
        phase: str,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Log one epoch of metrics.

        Args:
            epoch: Current epoch number.
            phase: 'train' or 'val'.
            metrics: Dict of metric names to values. Only COLUMNS keys are written.
        """
        row = {
            "epoch": epoch,
            "model": self.model_name,
            "scenario": self.scenario,
            "phase": phase,
            "timestamp": datetime.utcnow().isoformat(),
        }
        for col in self.COLUMNS:
            if col not in row:
                row[col] = metrics.get(col, "")

        if os.environ.get("LOCAL_RANK", "0") == "0":
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
                writer.writerow(row)

        if self.writer:
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    self.writer.add_scalar(f"{phase}/{key}", val, epoch)

    def close(self):
        if self.writer:
            self.writer.close()

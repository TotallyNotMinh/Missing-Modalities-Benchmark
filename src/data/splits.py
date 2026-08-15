import json
import random
from pathlib import Path
from typing import Dict, List, Optional


class SplitManager:
    """
    Manages the frozen 70/15/15% patient-wise train/val/test split for BraTS 2020.

    The split is generated once with a fixed seed and saved to a JSON file.
    All subsequent calls load from the same JSON file to guarantee reproducibility
    across all 8 models in the benchmark.
    """

    SPLITS = ["train", "val", "test"]

    def __init__(
        self,
        processed_dir: str,
        splits_file: str = "data/splits/splits.json",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        """
        Args:
            processed_dir: Path to the preprocessed patient directory.
                           Each subdirectory should be one patient ID.
            splits_file: Path to the JSON file where the split is persisted.
            train_ratio: Fraction of patients for training.
            val_ratio: Fraction of patients for validation.
            seed: Random seed for reproducibility.
        """
        self.processed_dir = Path(processed_dir)
        self.splits_file = Path(splits_file)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.seed = seed
        self._splits: Optional[Dict[str, List[str]]] = None

    def _discover_patients(self) -> List[str]:
        """Returns sorted list of patient IDs from the processed directory."""
        if not self.processed_dir.exists():
            raise FileNotFoundError(
                f"Processed directory not found: {self.processed_dir}. "
                "Run src/data/preprocess.py first."
            )
        patients = sorted([
            p.name for p in self.processed_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ])
        if not patients:
            raise ValueError(f"No patient directories found in {self.processed_dir}")
        return patients

    def generate(self, overwrite: bool = False) -> Dict[str, List[str]]:
        """
        Generates and saves the patient split.

        Args:
            overwrite: If True, regenerates even if splits_file already exists.

        Returns:
            Dict with keys 'train', 'val', 'test' mapping to lists of patient IDs.
        """
        if self.splits_file.exists() and not overwrite:
            print(f"[SplitManager] Split already exists at {self.splits_file}. Loading.")
            return self.load()

        patients = self._discover_patients()
        rng = random.Random(self.seed)
        rng.shuffle(patients)

        n = len(patients)
        n_train = int(round(n * self.train_ratio))
        n_val = int(round(n * self.val_ratio))

        splits = {
            "train": patients[:n_train],
            "val": patients[n_train:n_train + n_val],
            "test": patients[n_train + n_val:],
        }

        self.splits_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.splits_file, "w") as f:
            json.dump(splits, f, indent=2)

        print(
            f"[SplitManager] Split generated → {self.splits_file}\n"
            f"  Train: {len(splits['train'])} | "
            f"Val: {len(splits['val'])} | "
            f"Test: {len(splits['test'])}"
        )
        self._splits = splits
        return splits

    def load(self) -> Dict[str, List[str]]:
        """Loads the frozen split from disk."""
        if not self.splits_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {self.splits_file}. "
                "Run SplitManager.generate() first."
            )
        with open(self.splits_file, "r") as f:
            self._splits = json.load(f)
        return self._splits

    def get_split(self, split: str) -> List[str]:
        """
        Returns patient IDs for the requested split.

        Args:
            split: One of 'train', 'val', 'test'.
        """
        if split not in self.SPLITS:
            raise ValueError(f"Invalid split '{split}'. Choose from {self.SPLITS}.")
        if self._splits is None:
            self.load()
        return self._splits[split]

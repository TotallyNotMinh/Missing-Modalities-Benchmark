#!/usr/bin/env python3
"""
Generates splits_final.json in nnUNet_preprocessed/Dataset001_BraTS2020/
from the benchmark's data/splits/splits.json.

Fold 0 is configured to train on the benchmark's exact 70% train split
and validate on the benchmark's exact 15% validation split.
"""

import argparse
import json
from pathlib import Path


def generate_splits(splits_file: Path, preprocessed_dir: Path) -> None:
    if not splits_file.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_file}")

    with open(splits_file, "r") as f:
        splits = json.load(f)

    train_cases = sorted(splits["train"])
    val_cases = sorted(splits["val"])
    test_cases = sorted(splits["test"])

    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    splits_final_path = preprocessed_dir / "splits_final.json"

    # In nnU-Net, splits_final.json is a list of dicts with 'train' and 'val' keys.
    # Fold 0 = benchmark train / val split
    nnunet_splits = [
        {"train": train_cases, "val": val_cases}
    ]

    # Populate 4 additional synthetic folds for compatibility if 5-fold CV is ever requested
    combined = train_cases + val_cases
    n = len(combined)
    fold_size = n // 5
    for i in range(1, 5):
        val_fold = combined[i * fold_size : (i + 1) * fold_size]
        train_fold = [c for c in combined if c not in val_fold]
        nnunet_splits.append({"train": train_fold, "val": val_fold})

    with open(splits_final_path, "w") as f:
        json.dump(nnunet_splits, f, indent=2)

    print(f"[nnU-Net Splits] Saved splits_final.json to {splits_final_path}")
    print(f"  Fold 0 -> Train: {len(train_cases)} cases | Val: {len(val_cases)} cases")
    print(f"  (Test cases reserved strictly for benchmark evaluation: {len(test_cases)})")


def main():
    parser = argparse.ArgumentParser(description="Generate nnU-Net splits_final.json from benchmark splits")
    parser.add_argument("--splits_file", default="data/splits/splits.json", help="Path to benchmark splits.json")
    parser.add_argument("--preprocessed_dir", default="data/nnunet_preprocessed/Dataset001_BraTS2020", help="Path to preprocessed dataset dir")
    args = parser.parse_args()

    generate_splits(Path(args.splits_file), Path(args.preprocessed_dir))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Training entrypoint for nnU-Net v2 segmentation Oracle.

Trains the 4-channel 3D PlainConvUNet on 100% real complete MRI data
using the benchmark's standardized augmentation and data loading protocols.

Usage:
    python scripts/train_nnunet.py --epochs 100 --batch_size 2 --device cuda:0
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from tqdm import tqdm

from src.adapters.nnunet_adapter import nnUNetAdapter
from src.data.dataloader import get_dataloaders
from src.metrics.segmentation import compute_segmentation_metrics
from src.utils.checkpoint import CheckpointManager, EarlyStopping
from src.utils.logger import ExperimentLogger
from src.utils.pipeline_utils import load_config, seed_everything


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip: float = 1.0,
    use_amp: bool = True,
) -> float:
    """Executes one training epoch and returns mean training loss."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        # Full 4-channel stack: (B, 4, H, W, D)
        inputs = batch["modalities"].to(device, dtype=torch.float32)
        # Segmentation ground truth: (B, 1, H, W, D)
        targets = batch["mask"].to(device, dtype=torch.long)

        # Map BraTS label 4 to class index 3 so classes are {0, 1, 2, 3}
        target_mapped = targets.clone()
        target_mapped[target_mapped == 4] = 3

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            logits = model(inputs)
            loss = criterion(logits, target_mapped)

        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(1, num_batches)


@torch.no_grad()
def evaluate_validation(
    adapter: nnUNetAdapter,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Evaluates the model on the validation DataLoader and returns mean metrics."""
    adapter.network.eval()
    val_losses = []
    all_dice_wt = []
    all_dice_tc = []
    all_dice_et = []
    all_hd95_mean = []

    for batch in tqdm(val_loader, desc="Validation", leave=False):
        inputs = batch["modalities"].to(device, dtype=torch.float32)
        targets = batch["mask"].to(device, dtype=torch.long)

        target_mapped = targets.clone()
        target_mapped[target_mapped == 4] = 3

        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            logits = adapter.network(inputs)
            loss = criterion(logits, target_mapped)
            val_losses.append(loss.item())

        # Subregion segmentation metrics per patient
        batch_results = adapter.evaluate_batch(batch)
        for r in batch_results:
            all_dice_wt.append(r["Dice_WT"])
            all_dice_tc.append(r["Dice_TC"])
            all_dice_et.append(r["Dice_ET"])
            if not np.isnan(r["HD95_Mean"]):
                all_hd95_mean.append(r["HD95_Mean"])

    mean_wt = float(np.mean(all_dice_wt)) if all_dice_wt else 0.0
    mean_tc = float(np.mean(all_dice_tc)) if all_dice_tc else 0.0
    mean_et = float(np.mean(all_dice_et)) if all_dice_et else 0.0
    mean_dice = (mean_wt + mean_tc + mean_et) / 3.0
    mean_hd95 = float(np.mean(all_hd95_mean)) if all_hd95_mean else float("nan")

    return {
        "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
        "dice_wt": mean_wt,
        "dice_tc": mean_tc,
        "dice_et": mean_et,
        "dice_mean": mean_dice,
        "hd95_mean": mean_hd95,
    }


def main():
    parser = argparse.ArgumentParser(description="Train nnU-Net Oracle on BraTS 2020 (Real 4-Channel Data)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override training max epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override initial learning rate")
    parser.add_argument("--device", default=None, help="Device to use ('cuda', 'cuda:0', 'cpu')")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation frequency in epochs")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader num_workers")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    args = parser.parse_args()

    # 1. Load config
    cfg = load_config(args.config, model_name="nnunet")
    seed_everything(cfg.seed)

    # 2. Resolve hyperparams
    device_str = args.device or cfg.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print(f"[Training] CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    epochs = args.epochs or cfg.training.get("max_epochs", 100)
    batch_size = args.batch_size or cfg.training.get("batch_size", 2)
    lr = args.lr or cfg.training.get("learning_rate", 1e-4)
    num_workers = args.num_workers if args.num_workers is not None else cfg.get("num_workers", 4)
    patch_size = cfg.patch.get("size", [128, 128, 128])
    use_amp = cfg.training.get("amp", True) and (device.type == "cuda")
    grad_clip = cfg.training.get("gradient_clip", 1.0)
    val_interval = args.val_interval

    print(f"============================================================")
    print(f" Starting Oracle nnU-Net Training (Real 4-Channel MRI)")
    print(f" Device      : {device}")
    print(f" Epochs      : {epochs}")
    print(f" Batch Size  : {batch_size}")
    print(f" Learning Rate: {lr}")
    print(f" Patch Size  : {patch_size}")
    print(f" AMP         : {use_amp}")
    print(f"============================================================")

    # 3. Check splits existence
    splits_file = Path(cfg.paths.splits_file)
    if not splits_file.exists():
        print(f"[Training] Splits file {splits_file} not found. Generating now...")
        from src.data.splits import SplitManager
        manager = SplitManager(processed_dir=cfg.paths.preprocessed_cache, splits_file=str(splits_file))
        manager.generate()

    # 4. DataLoaders (load real 4-channel volumes)
    train_loader, val_loader, test_loader = get_dataloaders(
        scenario="S1",
        processed_dir=cfg.paths.preprocessed_cache,
        splits_file=str(splits_file),
        patch_size=patch_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=cfg.get("pin_memory", True) and (device.type == "cuda"),
        seed=cfg.seed,
        task="segmentation",
        cfg=cfg,
    )
    print(f"[Training] DataLoaders ready: {len(train_loader)} train batches, {len(val_loader)} val batches.")

    # 5. Model & Adapter
    adapter = nnUNetAdapter(
        device=device,
        patch_size=tuple(patch_size),
        num_input_channels=4,
        num_classes=4,
    )
    model = adapter.network

    # 6. Loss, Optimizer, Scheduler, AMP Scaler
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.training.get("weight_decay", 1e-5))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    # 7. Checkpoint & Logging (saves directly to checkpoints/oracle_nnunet)
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=cfg.paths.checkpoints_dir,
        model_name="oracle_nnunet",
        scenario="oracle",
    )
    early_stopping = EarlyStopping(
        patience=cfg.training.get("early_stopping_patience", 20),
        mode="max",
    )
    logger = ExperimentLogger(
        log_dir=cfg.paths.logs_dir,
        model_name="oracle_nnunet",
        scenario="oracle",
    )

    start_epoch = 1
    best_dice = 0.0

    if args.resume:
        ckpt_info = checkpoint_mgr.load(model, path=args.resume, optimizer=optimizer, device=str(device))
        start_epoch = ckpt_info.get("epoch", 0) + 1
        best_dice = ckpt_info.get("metric", 0.0)
        print(f"[Training] Resumed from epoch {start_epoch} (best metric={best_dice:.4f})")

    # 8. Main Training Loop
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            grad_clip=grad_clip,
            use_amp=use_amp,
        )
        scheduler.step()

        print(f"Epoch [{epoch:03d}/{epochs:03d}] Loss: {train_loss:.4f} LR: {scheduler.get_last_lr()[0]:.6f}")

        # Validation pass
        if epoch % val_interval == 0 or epoch == epochs:
            val_metrics = evaluate_validation(
                adapter=adapter,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
            )

            current_dice = val_metrics["dice_mean"]
            is_best = current_dice > best_dice
            if is_best:
                best_dice = current_dice

            print(f" -> Val Loss: {val_metrics['val_loss']:.4f} | "
                  f"Dice WT: {val_metrics['dice_wt']:.3f} | "
                  f"TC: {val_metrics['dice_tc']:.3f} | "
                  f"ET: {val_metrics['dice_et']:.3f} | "
                  f"Mean: {current_dice:.3f} (Best: {best_dice:.3f})")

            # Checkpointing
            checkpoint_mgr.save(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metric=current_dice,
                extra={"val_loss": val_metrics["val_loss"]},
                is_best=is_best,
            )

            # Logging
            logger.log_epoch(
                epoch=epoch,
                phase="val",
                loss=val_metrics["val_loss"],
                dice_wt=val_metrics["dice_wt"],
                dice_tc=val_metrics["dice_tc"],
                dice_et=val_metrics["dice_et"],
                dice_mean=current_dice,
                hd95_mean=val_metrics["hd95_mean"],
            )

            if early_stopping.step(current_dice):
                print(f"[Training] Early stopping triggered at epoch {epoch}.")
                break

    print(f"\n[Training] Complete! Best Val Dice: {best_dice:.4f}")
    print(f"[Training] Best weights saved at: {checkpoint_mgr.best_path}")


if __name__ == "__main__":
    main()

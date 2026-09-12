#!/usr/bin/env python3
"""
Training entrypoint for RFNet (Region-Aware Fusion Network) baseline on BraTS 2020.

Trains the RFNet architecture (~8.98M parameters, basic_dims=16)
using the benchmark's standardized splits.json, permutation from benchmark order
(T1, T1ce, T2, FLAIR) -> RFNet order (FLAIR, T1ce, T1, T2), on-the-fly 15-combination
missing modality dropout, and deep supervision auxiliary losses.

Usage:
    python scripts/train_rfnet.py --epochs 300 --batch_size 1 --device cuda:0
    python scripts/train_rfnet.py --smoke_test --device cpu
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure external RFNet is importable
RFNET_DIR = PROJECT_ROOT / "externals" / "rfnet"
if RFNET_DIR.exists() and str(RFNET_DIR) not in sys.path:
    sys.path.insert(0, str(RFNET_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import models
from utils import criterions

from functools import partial

from src.adapters.rfnet_adapter import RFNetAdapter
from src.data.brats_dataset import BraTSDataset
from src.data.splits import SplitManager
from src.data.augmentation import get_segmentation_train_transforms, get_val_transforms
from src.metrics.segmentation import compute_segmentation_metrics
from src.utils.pipeline_utils import load_config, seed_everything, worker_init_fn


# 15 combinatorial masks of [FLAIR, T1ce, T1, T2]
# True = available, False = missing
COMBINATORIAL_MASKS = np.array([
    [True, False, False, False],
    [False, True, False, False],
    [False, False, True, False],
    [False, False, False, True],
    [True, True, False, False],
    [True, False, True, False],
    [True, False, False, True],
    [False, True, True, False],
    [False, True, False, True],
    [False, False, True, True],
    [True, True, True, False],
    [True, True, False, True],
    [True, False, True, True],
    [False, True, True, True],
    [True, True, True, True],
], dtype=bool)

# Permutation indices: benchmark [T1, T1ce, T2, FLAIR] -> RFNet [FLAIR, T1ce, T1, T2]
BENCHMARK_TO_RFNET = [3, 1, 0, 2]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    region_fusion_start_epoch: int = 100,
    grad_clip: float = 1.0,
    grad_accum: int = 1,
    max_iters: Optional[int] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Executes one training epoch of RFNet with random combinatorial modality masking,
    auxiliary multi-head losses, and optional automatic mixed precision (AMP).
    """
    model.train()
    if hasattr(model, "module"):
        model.module.is_training = True
    else:
        model.is_training = True

    total_loss_sum = 0.0
    fuse_loss_sum = 0.0
    sep_loss_sum = 0.0
    prm_loss_sum = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)):
        if max_iters is not None and i >= max_iters:
            break

        # 1. Inputs: (B, 4, H, W, D) in benchmark order [T1, T1ce, T2, FLAIR]
        raw_modalities = batch["modalities"]
        # Permute to RFNet order [FLAIR, T1ce, T1, T2]
        inputs = raw_modalities[:, BENCHMARK_TO_RFNET, ...].to(device, dtype=torch.float32)
        b_size, _, H, W, D = inputs.shape

        # 2. Ground truth mask: (B, 1, H, W, D) with BraTS labels {0, 1, 2, 4}
        targets = batch["mask"].to(device, dtype=torch.long)
        # Remap label 4 -> 3 to give {0, 1, 2, 3}
        target_mapped = targets.clone()
        target_mapped[target_mapped == 4] = 3

        # One-hot encoding of target: (B, 4, H, W, D) required by RFNet's criterions.py
        target_one_hot = torch.zeros((b_size, 4, H, W, D), device=device, dtype=torch.float32)
        target_one_hot.scatter_(1, target_mapped, 1.0)

        # 3. Sample random modality mask from 15 combinations for each sample in batch
        mask_indices = np.random.choice(15, size=b_size)
        masks = torch.from_numpy(COMBINATORIAL_MASKS[mask_indices]).to(device=device, dtype=torch.bool)

        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            # 4. Model forward pass
            fuse_pred, sep_preds, prm_preds = model(inputs, masks)

            # 5. Losses
            # Fused prediction loss
            fuse_cross = criterions.softmax_weighted_loss(fuse_pred, target_one_hot, num_cls=4)
            fuse_dice = criterions.dice_loss(fuse_pred, target_one_hot, num_cls=4)
            fuse_loss = fuse_cross + fuse_dice

            # Separate modality encoder auxiliary loss
            sep_cross = torch.zeros(1, device=device, dtype=torch.float32)
            sep_dice = torch.zeros(1, device=device, dtype=torch.float32)
            for sep_pred in sep_preds:
                sep_cross += criterions.softmax_weighted_loss(sep_pred, target_one_hot, num_cls=4)
                sep_dice += criterions.dice_loss(sep_pred, target_one_hot, num_cls=4)
            sep_loss = sep_cross + sep_dice

            # Progressive region module deep supervision loss
            prm_cross = torch.zeros(1, device=device, dtype=torch.float32)
            prm_dice = torch.zeros(1, device=device, dtype=torch.float32)
            for prm_pred in prm_preds:
                prm_cross += criterions.softmax_weighted_loss(prm_pred, target_one_hot, num_cls=4)
                prm_dice += criterions.dice_loss(prm_pred, target_one_hot, num_cls=4)
            prm_loss = prm_cross + prm_dice

            # Total combined loss
            if epoch < region_fusion_start_epoch:
                loss = sep_loss + prm_loss
            else:
                loss = fuse_loss + sep_loss + prm_loss

            loss_for_backward = loss / max(1, grad_accum)

        if scaler is not None and use_amp and device.type == "cuda":
            scaler.scale(loss_for_backward).backward()
            if (num_batches + 1) % grad_accum == 0 or (num_batches + 1) == len(loader):
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss_for_backward.backward()
            if (num_batches + 1) % grad_accum == 0 or (num_batches + 1) == len(loader):
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

        total_loss_sum += loss.item()
        fuse_loss_sum += fuse_loss.item()
        sep_loss_sum += sep_loss.item()
        prm_loss_sum += prm_loss.item()
        num_batches += 1

    denom = max(1, num_batches)
    return {
        "loss": total_loss_sum / denom,
        "fuse_loss": fuse_loss_sum / denom,
        "sep_loss": sep_loss_sum / denom,
        "prm_loss": prm_loss_sum / denom,
    }


@torch.no_grad()
def evaluate_validation(
    adapter: RFNetAdapter,
    val_loader: DataLoader,
    max_cases: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluates the model on validation set using full sliding-window inference
    across the standard benchmark scenarios: Full modality and S1-S4.
    """
    adapter.network.eval()
    if hasattr(adapter.network, "is_training"):
        adapter.network.is_training = False

    all_dice_wt = []
    all_dice_tc = []
    all_dice_et = []
    all_hd95 = []

    case_count = 0
    for batch in tqdm(val_loader, desc="Validation", leave=False):
        if max_cases is not None and case_count >= max_cases:
            break

        inputs = batch["modalities"]
        targets = batch["mask"].squeeze(1).numpy()
        patient_ids = batch.get("patient_id", [f"val_{case_count}"])

        for b in range(inputs.shape[0]):
            sample_in = inputs[b]
            sample_gt = targets[b]

            metrics = adapter.evaluate_sample(
                input_data=sample_in,
                target_mask=sample_gt,
                mask="FULL",
                postprocess_et=True,
                et_threshold=500,
            )

            all_dice_wt.append(metrics["Dice_WT"])
            all_dice_tc.append(metrics["Dice_TC"])
            all_dice_et.append(metrics["Dice_ET"])
            all_hd95.append(metrics["HD95_Mean"])
            case_count += 1
            if max_cases is not None and case_count >= max_cases:
                break

    mean_wt = float(np.mean(all_dice_wt)) if all_dice_wt else 0.0
    mean_tc = float(np.mean(all_dice_tc)) if all_dice_tc else 0.0
    mean_et = float(np.mean(all_dice_et)) if all_dice_et else 0.0
    mean_dice = (mean_wt + mean_tc + mean_et) / 3.0
    mean_hd95 = float(np.mean(all_hd95)) if all_hd95 else 0.0

    return {
        "dice_mean": mean_dice,
        "dice_wt": mean_wt,
        "dice_tc": mean_tc,
        "dice_et": mean_et,
        "hd95_mean": mean_hd95,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="RFNet Baseline Training on BraTS 2020")
    parser.add_argument("--config", type=str, default="configs/models/rfnet.yaml", help="Path to rfnet.yaml config")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to preprocessed BraTS data")
    parser.add_argument("--splits_file", type=str, default=None, help="Path to splits.json")
    parser.add_argument("--save_dir", type=str, default="checkpoints/rfnet", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128], help="Training crop patch size")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--val_interval", type=int, default=5, help="Epoch interval between full evaluations")
    parser.add_argument("--seed", type=int, default=1024, help="Global random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1 epoch with 2 iterations for quick sanity check")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision (AMP)")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    cfg = load_config("config.yaml") if Path("config.yaml").exists() else None

    epochs = args.epochs or 300
    batch_size = args.batch_size or 1
    lr = args.lr or 1.0e-4
    patch_size = tuple(args.patch_size)
    device = torch.device(args.device)
    val_interval = 1 if args.smoke_test else args.val_interval
    use_amp = (not args.no_amp) and (cfg.get("amp", cfg.training.get("amp", True)) if cfg else True)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda")) if device.type == "cuda" else None

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = save_dir / "logs"
    writer = SummaryWriter(str(log_dir))

    print(f"============================================================")
    print(f" RFNet Training Pipeline Initializing")
    print(f" Device: {device} | Patch Size: {patch_size}")
    print(f" Batch Size: {batch_size} | Epochs: {epochs} | Initial LR: {lr}")
    print(f" Output Checkpoints: {save_dir}")
    print(f"============================================================")

    # 1. Dataset & DataLoaders
    data_dir = args.data_dir or (cfg.paths.preprocessed_cache if cfg else "data/processed")
    splits_file = Path(args.splits_file or (cfg.paths.splits_file if cfg else "data/splits/splits.json"))

    if not splits_file.exists():
        print(f"[RFNet] Splits file {splits_file} not found. Generating now...")
        manager = SplitManager(processed_dir=data_dir, splits_file=str(splits_file))
        manager.generate()

    split_mgr = SplitManager(processed_dir=data_dir, splits_file=str(splits_file))
    train_ids = split_mgr.get_split("train")
    val_ids = split_mgr.get_split("val")

    if args.smoke_test:
        train_ids = train_ids[:2]
        val_ids = val_ids[:1]
        epochs = 1

    train_transforms = get_segmentation_train_transforms(patch_size=patch_size)
    val_transforms = get_val_transforms()

    train_ds = BraTSDataset(data_dir=data_dir, patient_ids=train_ids, transform=train_transforms)
    val_ds = BraTSDataset(data_dir=data_dir, patient_ids=val_ids, transform=val_transforms)

    worker_init = partial(worker_init_fn, seed=args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if args.smoke_test else args.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=worker_init,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0 if args.smoke_test else min(2, args.num_workers),
        pin_memory=False,
    )

    # 2. Build Model
    model = models.Model(num_cls=4)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[RFNet] Model constructed with {total_params:,} parameters (~{total_params/1e6:.2f}M).")

    # 3. Optimizer & Polynomial LR Scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=1.0e-4,
        amsgrad=True,
    )

    def poly_lr(current_epoch: int) -> float:
        return (1.0 - current_epoch / max(1, epochs)) ** 0.9

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lr)

    # 4. Resume from Checkpoint
    start_epoch = 0
    best_dice = 0.0
    if args.resume is not None and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "optim_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optim_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice = ckpt.get("best_dice", 0.0)
        print(f"[RFNet] Resumed checkpoint from '{args.resume}' at epoch {start_epoch} (Best Dice: {best_dice:.4f}).")

    val_adapter = RFNetAdapter(network=model, device=device, patch_size=patch_size)

    # 5. Training Loop
    start_time = time.time()
    max_train_iters = 2 if args.smoke_test else None

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            region_fusion_start_epoch=100,
            grad_clip=args.grad_clip,
            grad_accum=args.grad_accum,
            max_iters=max_train_iters,
            scaler=scaler,
            use_amp=use_amp,
        )

        scheduler.step()

        writer.add_scalar("train/loss", train_metrics["loss"], global_step=epoch + 1)
        writer.add_scalar("train/fuse_loss", train_metrics["fuse_loss"], global_step=epoch + 1)
        writer.add_scalar("train/sep_loss", train_metrics["sep_loss"], global_step=epoch + 1)
        writer.add_scalar("train/prm_loss", train_metrics["prm_loss"], global_step=epoch + 1)
        writer.add_scalar("train/lr", current_lr, global_step=epoch + 1)

        epoch_duration = time.time() - epoch_start
        print(
            f"Epoch [{epoch+1}/{epochs}] ({epoch_duration:.1f}s) - "
            f"Loss: {train_metrics['loss']:.4f} "
            f"(Fuse: {train_metrics['fuse_loss']:.4f}, "
            f"Sep: {train_metrics['sep_loss']:.4f}, "
            f"PRM: {train_metrics['prm_loss']:.4f}) | "
            f"LR: {current_lr:.6f}"
        )

        # Save latest model checkpoint
        last_ckpt_path = save_dir / "model_last.pth"
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optim_dict": optimizer.state_dict(),
                "best_dice": best_dice,
            },
            last_ckpt_path,
        )

        # Validation evaluation
        if (epoch + 1) % val_interval == 0 or (epoch + 1) == epochs:
            print(f"[Validation] Evaluating epoch {epoch+1}...")
            val_metrics = evaluate_validation(
                adapter=val_adapter,
                val_loader=val_loader,
                max_cases=(1 if args.smoke_test else None),
            )

            mean_dice = val_metrics["dice_mean"]
            writer.add_scalar("val/dice_mean", mean_dice, global_step=epoch + 1)
            writer.add_scalar("val/dice_wt", val_metrics["dice_wt"], global_step=epoch + 1)
            writer.add_scalar("val/dice_tc", val_metrics["dice_tc"], global_step=epoch + 1)
            writer.add_scalar("val/dice_et", val_metrics["dice_et"], global_step=epoch + 1)

            print(
                f"[Validation] Epoch [{epoch+1}/{epochs}] - "
                f"Mean Dice: {mean_dice:.4f} "
                f"(WT: {val_metrics['dice_wt']:.4f}, "
                f"TC: {val_metrics['dice_tc']:.4f}, "
                f"ET: {val_metrics['dice_et']:.4f}) | "
                f"HD95: {val_metrics['hd95_mean']:.2f}"
            )

            # Save best model checkpoint
            if mean_dice > best_dice:
                best_dice = mean_dice
                best_ckpt_path = save_dir / "model_best.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optim_dict": optimizer.state_dict(),
                        "best_dice": best_dice,
                    },
                    best_ckpt_path,
                )
                print(f"[Validation] New best model saved to {best_ckpt_path} with Mean Dice: {best_dice:.4f}")

            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Periodic checkpoint
        if (epoch + 1) % 50 == 0:
            periodic_path = save_dir / f"model_epoch_{epoch+1}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optim_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                },
                periodic_path,
            )

    writer.close()
    total_time_hours = (time.time() - start_time) / 3600.0
    print(f"============================================================")
    if args.smoke_test:
        print(f" [SMOKE TEST PASSED] RFNet pipeline verified successfully!")
    else:
        print(f" Training complete in {total_time_hours:.2f} hours.")
        print(f" Best Validation Mean Dice: {best_dice:.4f}")
    print(f" Checkpoints saved in: {save_dir}")
    print(f"============================================================")


if __name__ == "__main__":
    main()

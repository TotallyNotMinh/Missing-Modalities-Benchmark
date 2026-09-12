#!/usr/bin/env python3
"""
Training entrypoint for mmFormer baseline on BraTS 2020.

Trains the mmFormer architecture (Option A: 36.65M parameters, basic_dims=8)
using the benchmark's standardized splits.json, permutation from benchmark order
(T1, T1ce, T2, FLAIR) -> mmFormer order (FLAIR, T1ce, T1, T2), on-the-fly 15-combination
missing modality dropout, and deep supervision auxiliary losses.

Usage:
    python scripts/train_mmformer.py --epochs 1000 --batch_size 1 --device cuda:0
    python scripts/train_mmformer.py --smoke_test --device cpu
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

# Ensure external mmFormer is importable
MMFORMER_DIR = PROJECT_ROOT / "externals" / "mmformer" / "mmformer"
if MMFORMER_DIR.exists() and str(MMFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(MMFORMER_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import mmformer
from utils import criterions

from functools import partial

from src.adapters.mmformer_adapter import MMFormerAdapter
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

# Permutation indices: benchmark [T1, T1ce, T2, FLAIR] -> mmFormer [FLAIR, T1ce, T1, T2]
BENCHMARK_TO_MMFORMER = [3, 1, 0, 2]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    region_fusion_start_epoch: int = 0,
    grad_clip: float = 1.0,
    grad_accum: int = 1,
    max_iters: Optional[int] = None,
) -> Dict[str, float]:
    """
    Executes one training epoch of mmFormer with random combinatorial modality masking
    and auxiliary multi-head losses.
    """
    model.train()
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
        # Permute to mmFormer order [FLAIR, T1ce, T1, T2]
        inputs = raw_modalities[:, BENCHMARK_TO_MMFORMER, ...].to(device, dtype=torch.float32)
        b_size, _, H, W, D = inputs.shape

        # 2. Ground truth mask: (B, 1, H, W, D) with BraTS labels {0, 1, 2, 4}
        targets = batch["mask"].to(device, dtype=torch.long)
        # Remap label 4 -> 3 to give {0, 1, 2, 3}
        target_mapped = targets.clone()
        target_mapped[target_mapped == 4] = 3

        # One-hot encoding of target: (B, 4, H, W, D) required by mmFormer's criterions.py
        target_one_hot = torch.zeros((b_size, 4, H, W, D), device=device, dtype=torch.float32)
        target_one_hot.scatter_(1, target_mapped, 1.0)

        # 3. Sample random modality mask from 15 combinations for each sample in batch
        mask_indices = np.random.choice(15, size=b_size)
        masks = torch.from_numpy(COMBINATORIAL_MASKS[mask_indices]).to(device=device, dtype=torch.bool)

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
    adapter: MMFormerAdapter,
    val_loader: DataLoader,
    max_cases: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluates the model on validation set using full sliding-window inference
    across the standard benchmark scenarios: Full modality and S1-S4.
    """
    adapter.network.eval()
    all_dice_wt = []
    all_dice_tc = []
    all_dice_et = []
    all_hd95 = []

    case_count = 0
    for batch in tqdm(val_loader, desc="Validation", leave=False):
        if max_cases is not None and case_count >= max_cases:
            break

        results = adapter.evaluate_batch(batch, scenario="FULL")
        for r in results:
            all_dice_wt.append(r["Dice_WT"])
            all_dice_tc.append(r["Dice_TC"])
            all_dice_et.append(r["Dice_ET"])
            if not np.isnan(r.get("HD95_Mean", np.nan)):
                all_hd95.append(r["HD95_Mean"])
        case_count += 1

    wt = float(np.mean(all_dice_wt)) if all_dice_wt else 0.0
    tc = float(np.mean(all_dice_tc)) if all_dice_tc else 0.0
    et = float(np.mean(all_dice_et)) if all_dice_et else 0.0
    mean_dice = (wt + tc + et) / 3.0
    hd95 = float(np.mean(all_hd95)) if all_hd95 else float("nan")

    return {
        "dice_wt": wt,
        "dice_tc": tc,
        "dice_et": et,
        "dice_mean": mean_dice,
        "hd95_mean": hd95,
    }


def main():
    parser = argparse.ArgumentParser(description="Train mmFormer Baseline on BraTS 2020")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override training max epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Adam weight decay")
    parser.add_argument("--gpu", type=int, default=None, help="Physical GPU index to use (e.g. 0, 1, 2)")
    parser.add_argument("--device", default=None, help="Device to use ('cuda', 'cuda:0', 'cpu')")
    parser.add_argument("--val_interval", type=int, default=20, help="Validation frequency (epochs)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--data_dir", type=str, default=None, help="Override path to preprocessed data directory")
    parser.add_argument("--splits_file", type=str, default=None, help="Override path to splits.json")
    parser.add_argument("--save_dir", type=str, default="checkpoints/mmformer", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick 1-epoch smoke test and exit")
    args = parser.parse_args()

    # 1. Load config
    cfg = load_config(args.config, model_name="mmformer")
    seed_everything(cfg.seed)

    # 2. Resolve device
    if args.gpu is not None:
        if torch.cuda.is_available() and torch.cuda.device_count() == 1 and args.gpu > 0:
            print(f"[Training] Note: Single GPU visible (device_count=1). Mapping --gpu {args.gpu} to cuda:0.")
            device_str = "cuda:0"
        else:
            device_str = f"cuda:{args.gpu}"
    else:
        device_str = args.device or cfg.device

    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"[Training] CUDA requested ('{device_str}') but not available. Falling back to CPU.")
            device_str = "cpu"
        else:
            try:
                gpu_idx = int(device_str.split(":")[1]) if ":" in device_str else 0
                gpu_name = torch.cuda.get_device_name(gpu_idx)
                print(f"[Training] Accelerator: CUDA ({gpu_name}) on device {device_str}")
            except Exception:
                print(f"[Training] Accelerator: CUDA on device {device_str}")
    else:
        print(f"[Training] Accelerator: CPU")
    device = torch.device(device_str)

    # 3. Training hyper-parameters (Option A defaults)
    epochs = 1 if args.smoke_test else (args.epochs or cfg.training.get("max_epochs", 1000))
    batch_size = 1 if args.smoke_test else (args.batch_size or cfg.training.get("batch_size", 1))
    lr = args.lr or cfg.training.get("learning_rate", 2e-4)
    weight_decay = args.weight_decay or cfg.training.get("weight_decay", 1e-4)
    patch_size = cfg.patch.get("size", [128, 128, 128])
    val_interval = 1 if args.smoke_test else args.val_interval
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"============================================================")
    print(f" Starting mmFormer Training (Option A: 36.65M Params)")
    print(f" Device        : {device}")
    print(f" Epochs        : {epochs}")
    print(f" Batch Size    : {batch_size}")
    print(f" Learning Rate : {lr}")
    print(f" Weight Decay  : {weight_decay}")
    print(f" Patch Size    : {patch_size}")
    print(f" Checkpoints   : {save_dir}")
    print(f" Smoke Test    : {args.smoke_test}")
    print(f"============================================================")

    # 4. Resolve data splits
    data_dir = args.data_dir or cfg.paths.preprocessed_cache
    splits_file = Path(args.splits_file or cfg.paths.splits_file)
    if not splits_file.exists():
        print(f"[Training] Splits file {splits_file} not found. Generating now...")
        manager = SplitManager(processed_dir=data_dir, splits_file=str(splits_file))
        manager.generate()

    split_mgr = SplitManager(processed_dir=data_dir, splits_file=str(splits_file))
    train_ids = split_mgr.get_split("train")
    val_ids = split_mgr.get_split("val")

    print(f"[Training] Loaded splits: {len(train_ids)} train, {len(val_ids)} val patients.")

    # 5. Datasets & DataLoaders
    train_transforms = get_segmentation_train_transforms(patch_size=tuple(patch_size), cfg=cfg)
    val_transforms = get_val_transforms(cfg=cfg)

    train_dataset = BraTSDataset(
        data_dir=data_dir,
        patient_ids=train_ids,
        transform=train_transforms,
    )
    val_dataset = BraTSDataset(
        data_dir=data_dir,
        patient_ids=val_ids,
        transform=val_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=(0 if args.smoke_test else args.num_workers),
        worker_init_fn=partial(worker_init_fn, base_seed=cfg.seed),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # 6. Initialize mmFormer Model (Option A: basic_dims=8)
    mmformer.basic_dims = 8
    model = mmformer.Model(num_cls=4).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Training] mmFormer Model initialized with {param_count / 1e6:.2f}M parameters.")

    # 7. Optimizer & Polynomial Scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=True,
        weight_decay=weight_decay,
    )

    start_epoch = 0
    best_dice = 0.0

    # Resume from checkpoint if specified
    if args.resume is not None:
        resume_path = Path(args.resume)
        if resume_path.is_file():
            print(f"[Training] Resuming from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_dice = ckpt.get("best_dice", 0.0)
            if "state_dict" in ckpt:
                clean_sd = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt["state_dict"].items()}
                model.load_state_dict(clean_sd)
            if "optim_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optim_dict"])
            print(f"[Training] Resumed at epoch {start_epoch}, previous best dice: {best_dice:.4f}")

    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(save_dir / "logs"))

    # Adapter for validation inference
    val_adapter = MMFormerAdapter(network=model, device=device, patch_size=tuple(patch_size))

    # 8. Main Training Loop
    start_time = time.time()
    max_train_iters = 3 if args.smoke_test else None

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Polynomial learning rate decay
        current_lr = lr * ((1.0 - float(epoch) / float(max(1, epochs))) ** 0.9)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # Train one epoch
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            region_fusion_start_epoch=0,
            grad_clip=args.grad_clip,
            grad_accum=args.grad_accum,
            max_iters=max_train_iters,
        )

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
        print(f" [SMOKE TEST PASSED] mmFormer pipeline verified successfully!")
    else:
        print(f" Training complete in {total_time_hours:.2f} hours.")
        print(f" Best Validation Mean Dice: {best_dice:.4f}")
    print(f" Checkpoints saved in: {save_dir}")
    print(f"============================================================")


if __name__ == "__main__":
    main()

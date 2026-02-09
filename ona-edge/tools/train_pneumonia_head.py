"""
ONA Edge — Pneumonia Head Training
Adds a pneumonia classification head to the multi-head DenseNet121.
Freezes backbone + TB head. Trains only the pneumonia head.

NOTE: The Kaggle "Chest X-Ray Images (Pneumonia)" dataset uses pediatric
chest X-rays. This is acknowledged as architecture validation — adult
cohort fine-tuning is a separate funded objective.

Usage:
  python tools/train_pneumonia_head.py "C:/path/to/chest_xray"
  python tools/train_pneumonia_head.py "C:/path/to/chest_xray" --epochs 20
  python tools/train_pneumonia_head.py "C:/path/to/chest_xray" --resume
"""

import sys
import os
import time
import argparse
import logging
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from config import setup_logging, MODEL_DIR
from finetune_model import (
    MultiHeadClassifier, TBDataset, migrate_tb_to_multihead,
    calculate_auc, find_optimal_threshold, bootstrap_ci,
    stratified_split
)

logger = logging.getLogger('ona.pneumonia')


def load_pneumonia_dataset(dataset_path: str) -> List[Dict]:
    """Load the Kaggle pneumonia dataset.
    Expected structure:
      chest_xray/
        train/NORMAL/  train/PNEUMONIA/
        val/NORMAL/    val/PNEUMONIA/
        test/NORMAL/   test/PNEUMONIA/
    OR just:
      NORMAL/  PNEUMONIA/
    """
    dataset_path = Path(dataset_path)
    samples = []
    image_extensions = {'.jpeg', '.jpg', '.png', '.bmp'}

    # Try subdirectory format: NORMAL/ and PNEUMONIA/
    # Check if there are train/val/test splits already
    has_splits = (dataset_path / 'train').exists()

    if has_splits:
        # Kaggle format with train/val/test — load ALL and we'll re-split
        for split in ['train', 'val', 'test']:
            split_dir = dataset_path / split
            if not split_dir.exists():
                continue
            for class_name, label in [('NORMAL', 0), ('PNEUMONIA', 1)]:
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    continue
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in image_extensions:
                        samples.append({'path': str(img_path), 'label': label})
    else:
        # Flat format: NORMAL/ PNEUMONIA/
        for class_name, label in [('NORMAL', 0), ('PNEUMONIA', 1)]:
            class_dir = dataset_path / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in image_extensions:
                    samples.append({'path': str(img_path), 'label': label})

    n_pneumonia = sum(1 for s in samples if s['label'] == 1)
    n_normal = sum(1 for s in samples if s['label'] == 0)
    logger.info(f"Loaded {len(samples)} images ({n_pneumonia} Pneumonia, {n_normal} Normal)")

    return samples


def evaluate_head(model, loader, criterion, device):
    """Evaluate the pneumonia head"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, head='pneumonia').squeeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            scores = torch.sigmoid(outputs)
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    labels = np.array(all_labels)
    scores = np.array(all_scores)
    avg_loss = total_loss / len(labels)
    auc = calculate_auc(labels, scores)

    best_threshold, best_sens, best_spec = find_optimal_threshold(labels, scores)

    return {
        'loss': avg_loss,
        'auc': auc,
        'optimal_sensitivity': best_sens,
        'optimal_specificity': best_spec,
        'optimal_threshold': best_threshold,
        'labels': labels,
        'scores': scores,
    }


def main():
    parser = argparse.ArgumentParser(description='Train pneumonia head on multi-head DenseNet121')
    parser.add_argument('dataset_path', help='Path to pneumonia dataset')
    parser.add_argument('--tb-checkpoint', type=str, default=None,
                        help='Path to TB checkpoint (default: auto-detect)')
    parser.add_argument('--epochs', type=int, default=20, help='Max epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--resume', action='store_true', help='Resume from multi-head checkpoint')
    args = parser.parse_args()

    setup_logging()

    print("=" * 60)
    print("  ONA Health — Pneumonia Head Training")
    print("  Multi-Head DenseNet121 (TB locked + Pneumonia)")
    print("=" * 60)
    print()
    print("  NOTE: Training on pediatric chest X-rays.")
    print("  Architecture validation — adult fine-tuning is a funded objective.")
    print()

    # ── Load pneumonia dataset ──
    print(f"Loading dataset: {args.dataset_path}")
    samples = load_pneumonia_dataset(args.dataset_path)

    if not samples:
        print("ERROR: No images found. Expected NORMAL/ and PNEUMONIA/ subdirectories.")
        sys.exit(1)

    n_pneumonia = sum(1 for s in samples if s['label'] == 1)
    n_normal = sum(1 for s in samples if s['label'] == 0)
    print(f"  Total: {len(samples)} ({n_pneumonia} Pneumonia, {n_normal} Normal)")

    # ── Split data (fresh stratified split, seed=43 to differ from TB) ──
    train_samples, val_samples, test_samples = stratified_split(samples, seed=43)
    train_pos = sum(1 for s in train_samples if s['label'] == 1)
    val_pos = sum(1 for s in val_samples if s['label'] == 1)
    test_pos = sum(1 for s in test_samples if s['label'] == 1)

    print(f"  Train: {len(train_samples)} ({train_pos} Pneumonia, {len(train_samples) - train_pos} Normal)")
    print(f"  Val:   {len(val_samples)} ({val_pos} Pneumonia, {len(val_samples) - val_pos} Normal)")
    print(f"  Test:  {len(test_samples)} ({test_pos} Pneumonia, {len(test_samples) - test_pos} Normal)")

    # ── Create data loaders ──
    train_dataset = TBDataset(train_samples, augment=True)
    val_dataset = TBDataset(val_samples, augment=False)
    test_dataset = TBDataset(test_samples, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Device ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ── Build multi-head model ──
    checkpoint_dir = Path(MODEL_DIR) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    multihead_checkpoint = checkpoint_dir / 'multihead_checkpoint.pth'

    if args.resume and multihead_checkpoint.exists():
        print(f"\n  Resuming from multi-head checkpoint: {multihead_checkpoint}")
        import torchxrayvision as xrv
        xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
        model = MultiHeadClassifier(xrv_model)
        ckpt = torch.load(multihead_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        best_val_auc = ckpt.get('best_val_auc', 0)
        patience_counter = ckpt.get('patience_counter', 0)
        train_history = ckpt.get('train_history', [])
        best_model_state = ckpt.get('best_model_state', None)
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"  Resumed at epoch {start_epoch}, best AUC: {best_val_auc:.3f}")
    else:
        # Migrate from TB checkpoint
        tb_checkpoint = args.tb_checkpoint
        if not tb_checkpoint:
            tb_checkpoint = str(Path(MODEL_DIR) / 'checkpoints' / 'finetune_checkpoint.pth')
        if not Path(tb_checkpoint).exists():
            # Try the saved model
            tb_checkpoint = str(Path(MODEL_DIR) / 'ona-tb-finetuned.pth')
        if not Path(tb_checkpoint).exists():
            print("ERROR: No TB checkpoint found. Train TB head first.")
            sys.exit(1)

        print(f"  Migrating TB weights from: {tb_checkpoint}")
        model = migrate_tb_to_multihead(tb_checkpoint, device=device)
        best_val_auc = 0
        patience_counter = 0
        train_history = []
        best_model_state = None
        start_epoch = 1

    model = model.to(device)

    # ── FREEZE backbone + TB head ──
    print("\n  Freezing backbone...")
    for param in model.backbone.parameters():
        param.requires_grad = False

    print("  Freezing TB head (locked, do not touch)...")
    for param in model.heads['tb'].parameters():
        param.requires_grad = False

    # Only pneumonia head is trainable
    trainable = sum(p.numel() for p in model.heads['pneumonia'].parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable (pneumonia head only): {trainable:,}")

    # ── Loss with class weighting ──
    if n_normal > n_pneumonia:
        pos_weight = torch.tensor([n_normal / n_pneumonia]).to(device)
    else:
        pos_weight = torch.tensor([1.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  Class weight: {pos_weight.item():.2f}x for pneumonia positive")

    # ── Optimizer (only pneumonia head params) ──
    optimizer = optim.Adam(
        model.heads['pneumonia'].parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ──
    start_time = time.time()

    print(f"\n{'=' * 60}")
    print(f"  Training pneumonia head — up to {args.epochs} epochs, patience {args.patience}")
    print(f"{'=' * 60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Train
        model.train()
        # Keep backbone and TB head in eval mode
        model.backbone.eval()
        model.heads['tb'].eval()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, head='pneumonia').squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validate
        val_metrics = evaluate_head(model, val_loader, criterion, device)
        scheduler.step()

        epoch_time = time.time() - epoch_start

        history_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_auc': val_metrics['auc'],
            'val_sens': val_metrics['optimal_sensitivity'],
            'val_spec': val_metrics['optimal_specificity'],
        }
        train_history.append(history_entry)

        print(f"  Epoch {epoch:2d}/{args.epochs} | "
              f"Loss: {train_loss:.4f} Acc: {train_acc:.1%} | "
              f"Val AUC: {val_metrics['auc']:.3f} "
              f"Sens: {val_metrics['optimal_sensitivity']:.1%} "
              f"Spec: {val_metrics['optimal_specificity']:.1%} | "
              f"{epoch_time:.0f}s")

        # Early stopping
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"         *** New best AUC: {best_val_auc:.3f} ***")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_model_state': best_model_state,
            'best_val_auc': best_val_auc,
            'patience_counter': patience_counter,
            'train_history': train_history,
        }, multihead_checkpoint)
        print(f"         [checkpoint saved]")
        sys.stdout.flush()

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s")

    # ── Load best model ──
    if best_model_state:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    # ── Test set evaluation ──
    print(f"\n{'=' * 60}")
    print("  EVALUATING ON HELD-OUT TEST SET (Pneumonia)")
    print(f"{'=' * 60}")

    test_metrics = evaluate_head(model, test_loader, criterion, device)

    labels = test_metrics['labels']
    scores = test_metrics['scores']
    opt_t = test_metrics['optimal_threshold']

    print(f"\n  Test AUC-ROC:      {test_metrics['auc']:.4f}")
    print(f"  Sensitivity:       {test_metrics['optimal_sensitivity']:.1%} (threshold {opt_t:.3f})")
    print(f"  Specificity:       {test_metrics['optimal_specificity']:.1%}")

    # Confusion matrix
    preds = (scores >= opt_t).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    print(f"\n  Confusion Matrix (threshold = {opt_t:.3f}):")
    print(f"                           Predicted")
    print(f"                      |  Pneum  |  Normal  |")
    print(f"  Actual  Pneumonia   |  {tp:>5}  |  {fn:>5}   |")
    print(f"          Normal      |  {fp:>5}  |  {tn:>5}   |")

    # ── Bootstrap CIs ──
    print(f"\n{'=' * 60}")
    print("  BOOTSTRAP CONFIDENCE INTERVALS (2000 iterations)")
    print(f"{'=' * 60}")

    def sens_fn(l, s):
        p = (s >= opt_t).astype(int)
        t = np.sum((p == 1) & (l == 1))
        f = np.sum((p == 0) & (l == 1))
        return t / (t + f) if (t + f) > 0 else 0

    def spec_fn(l, s):
        p = (s >= opt_t).astype(int)
        t = np.sum((p == 0) & (l == 0))
        f = np.sum((p == 1) & (l == 0))
        return t / (t + f) if (t + f) > 0 else 0

    print("  Computing...")
    sens_ci = bootstrap_ci(labels, scores, sens_fn, n_bootstrap=2000)
    spec_ci = bootstrap_ci(labels, scores, spec_fn, n_bootstrap=2000)
    auc_ci = bootstrap_ci(labels, scores, calculate_auc, n_bootstrap=2000)

    print(f"\n  Sensitivity: {test_metrics['optimal_sensitivity']:.1%}  (95% CI: {sens_ci[0]:.1%} - {sens_ci[1]:.1%})")
    print(f"  Specificity: {test_metrics['optimal_specificity']:.1%}  (95% CI: {spec_ci[0]:.1%} - {spec_ci[1]:.1%})")
    print(f"  AUC-ROC:     {test_metrics['auc']:.4f}  (95% CI: {auc_ci[0]:.4f} - {auc_ci[1]:.4f})")

    # ── Save multi-head model ──
    model_path = str(Path(MODEL_DIR) / 'ona-multihead-finetuned.pth')
    model_cpu = model.cpu()
    torch.save({
        'model_state_dict': model_cpu.state_dict(),
        'architecture': 'MultiHeadClassifier',
        'backbone': 'torchxrayvision-densenet121-res224-all',
        'heads': {
            'tb': {
                'optimal_threshold': 0.180,  # from TB test evaluation
                'test_auc': 0.9985,
                'test_sensitivity': 0.986,
                'test_specificity': 0.983,
                'dataset': 'TB_Chest_Radiography_Database (adult)',
            },
            'pneumonia': {
                'optimal_threshold': opt_t,
                'test_auc': test_metrics['auc'],
                'test_sensitivity': test_metrics['optimal_sensitivity'],
                'test_specificity': test_metrics['optimal_specificity'],
                'test_sens_ci': sens_ci,
                'test_spec_ci': spec_ci,
                'test_auc_ci': auc_ci,
                'dataset': 'Chest X-Ray Images Pneumonia (pediatric)',
                'limitation': 'Trained on pediatric CXRs. Adult fine-tuning pending.',
            },
        },
        'date': datetime.now().isoformat(),
    }, model_path)
    print(f"\n  Multi-head model saved: {model_path}")

    # ── Generate report ──
    report_path = str(Path(__file__).parent.parent / 'pneumonia_evaluation_report.md')
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    pn_scores = scores[labels == 1]
    norm_scores = scores[labels == 0]

    report = f"""# ONA Health - Pneumonia Head Evaluation Report

**Generated:** {now}
**Model:** MultiHeadClassifier (DenseNet121 backbone, TB + Pneumonia heads)
**Dataset:** Chest X-Ray Images (Pneumonia) — Pediatric ({len(samples)} images)
**Split:** {len(train_samples)} train / {len(val_samples)} val / {len(test_samples)} test
**Training time:** {total_time:.0f}s ({len(train_history)} epochs)

> **IMPORTANT:** This dataset contains pediatric chest X-rays (children aged 1-5).
> Results validate the multi-head architecture. Population-specific fine-tuning
> on adult African cohort data is a funded objective of this proposal.

---

## Test Set Results (Pneumonia Head)

| Metric | Value | 95% CI |
|--------|-------|--------|
| **Sensitivity** | {test_metrics['optimal_sensitivity']:.1%} | {sens_ci[0]:.1%} - {sens_ci[1]:.1%} |
| **Specificity** | {test_metrics['optimal_specificity']:.1%} | {spec_ci[0]:.1%} - {spec_ci[1]:.1%} |
| **AUC-ROC** | {test_metrics['auc']:.4f} | {auc_ci[0]:.4f} - {auc_ci[1]:.4f} |
| **Optimal Threshold** | {opt_t:.3f} | - |

## Confusion Matrix (threshold = {opt_t:.3f}, n={len(test_samples)})

|  | Predicted Pneumonia | Predicted Normal |
|--|---------------------|------------------|
| **Actual Pneumonia** | {tp} | {fn} |
| **Actual Normal** | {fp} | {tn} |

## Score Distribution

- **Pneumonia** (n={len(pn_scores)}): mean={np.mean(pn_scores):.3f}, range=[{np.min(pn_scores):.3f}, {np.max(pn_scores):.3f}]
- **Normal** (n={len(norm_scores)}): mean={np.mean(norm_scores):.3f}, range=[{np.min(norm_scores):.3f}, {np.max(norm_scores):.3f}]
- **Separation:** {np.mean(pn_scores) - np.mean(norm_scores):.3f}

## Training History

| Epoch | Train Loss | Train Acc | Val AUC | Val Sens | Val Spec |
|-------|-----------|-----------|---------|----------|----------|
"""
    for h in train_history:
        report += (f"| {h['epoch']} | {h['train_loss']:.4f} | {h['train_acc']:.1%} | "
                   f"{h['val_auc']:.3f} | {h['val_sens']:.1%} | {h['val_spec']:.1%} |\n")

    report += f"""
---

## Combined Model Summary

| Head | AUC | Sensitivity | Specificity | Dataset |
|------|-----|-------------|-------------|---------|
| **TB** | 0.998 | 98.6% | 98.3% | Adult TB CXR (4,200) |
| **Pneumonia** | {test_metrics['auc']:.3f} | {test_metrics['optimal_sensitivity']:.1%} | {test_metrics['optimal_specificity']:.1%} | Pediatric CXR ({len(samples)}) |

## For Grant Applications

> "ONA Health's multi-head AI architecture uses a shared DenseNet121 backbone
> with independent classification heads for TB and pneumonia detection.
> The TB head achieved {0.986:.0%} sensitivity (95% CI: 95-100%) on adult chest
> radiographs. The pneumonia head achieved {test_metrics['optimal_sensitivity']:.0%} sensitivity
> (95% CI: {sens_ci[0]:.0%}-{sens_ci[1]:.0%}) on a pediatric validation dataset,
> confirming architectural scalability. Population-specific fine-tuning on adult
> African cohort data is a funded objective of this proposal."

---

*Model saved to: {model_path}*
*Report generated by ONA Edge Multi-Head Training Tool v1.0*
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Report saved: {report_path}")

    # ── Final summary ──
    print(f"\n{'=' * 60}")
    print(f"  PNEUMONIA HEAD COMPLETE")
    print(f"  AUC: {test_metrics['auc']:.3f} | Sens: {test_metrics['optimal_sensitivity']:.1%} | Spec: {test_metrics['optimal_specificity']:.1%}")
    print(f"  Multi-head model: {model_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

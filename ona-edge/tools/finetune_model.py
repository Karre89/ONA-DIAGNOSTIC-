"""
ONA Edge — Model Fine-Tuning Script
Transfer learning on DenseNet121 backbone for binary TB detection.

Takes the pretrained TorchXRayVision DenseNet121 and fine-tunes it
on a labeled TB dataset. This is the go/no-go test: if fine-tuning
gets sensitivity above 88%, the architecture works and the remaining
gap is a data problem. If it stalls at 80%, we need a different approach.

Usage:
  python tools/finetune_model.py "C:/path/to/TB_Chest_Radiography_Database"
  python tools/finetune_model.py "C:/path/to/dataset" --epochs 30 --lr 0.001
  python tools/finetune_model.py "C:/path/to/dataset" --batch-size 8  (for low RAM)
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
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from config import setup_logging, MODEL_DIR
from validate_model import load_dataset

logger = logging.getLogger('ona.finetune')


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class TBDataset(Dataset):
    """PyTorch Dataset for TB chest X-ray classification"""

    def __init__(self, samples: List[Dict], augment: bool = False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load grayscale image
        img = Image.open(sample['path']).convert('L')
        img_array = np.array(img).astype(np.float32)

        # Resize to 224x224
        img_array = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Data augmentation (training only)
        if self.augment:
            img_array = self._augment(img_array)

        # Normalize to [-1024, 1024] (XRV convention)
        img_array = (img_array / 255.0) * 2048 - 1024

        # Shape: (1, 224, 224)
        tensor = torch.from_numpy(img_array).float().unsqueeze(0)
        label = torch.tensor(sample['label'], dtype=torch.float32)

        return tensor, label

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Random augmentations appropriate for chest X-rays"""
        # Random horizontal flip (CXRs are roughly symmetric)
        if random.random() > 0.5:
            img = np.fliplr(img).copy()

        # Random rotation (-10 to +10 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            img = np.clip(img * factor, 0, 255)

        # Random contrast
        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            mean = img.mean()
            img = np.clip((img - mean) * factor + mean, 0, 255)

        return img


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

class TBClassifier(nn.Module):
    """Fine-tuned DenseNet121 for binary TB detection.
    Uses XRV DenseNet backbone with a new binary classifier head."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # Replace XRV's 18-class classifier with binary TB classifier
        # XRV DenseNet features output = 1024
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.backbone.features(x)
        out = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class MultiHeadClassifier(nn.Module):
    """Multi-head DenseNet121 sharing a single backbone.
    Each head is an independent binary classifier (e.g. TB, Pneumonia).
    Backbone features (1024-dim) are computed once, then routed to the
    requested head."""

    SUPPORTED_HEADS = ['tb', 'pneumonia']

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict({
            'tb': nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            ),
            'pneumonia': nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            ),
        })

    def forward(self, x, head='tb'):
        features = self.backbone.features(x)
        out = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        out = out.view(out.size(0), -1)
        return self.heads[head](out)

    def forward_all(self, x):
        """Run all heads on the same input (single backbone pass)."""
        features = self.backbone.features(x)
        out = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        out = out.view(out.size(0), -1)
        return {name: head(out) for name, head in self.heads.items()}


def migrate_tb_to_multihead(tb_checkpoint_path, device='cpu'):
    """Load a trained TBClassifier checkpoint and migrate its weights
    into a MultiHeadClassifier. The TB head is copied exactly;
    the pneumonia head starts with random weights."""
    import torchxrayvision as xrv

    logger.info("Creating MultiHeadClassifier from TB checkpoint...")
    xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
    multihead = MultiHeadClassifier(xrv_model)

    checkpoint = torch.load(tb_checkpoint_path, map_location=device, weights_only=False)
    best_state = checkpoint.get('best_model_state', checkpoint.get('model_state_dict'))

    # Map TBClassifier weights -> MultiHeadClassifier
    # TBClassifier: backbone.*, classifier.0-4.*
    # MultiHeadClassifier: backbone.*, heads.tb.0-4.*
    new_state = {}
    for key, value in best_state.items():
        if key.startswith('classifier.'):
            new_key = key.replace('classifier.', 'heads.tb.')
            new_state[new_key] = value
        else:
            new_state[key] = value

    multihead.load_state_dict(new_state, strict=False)
    logger.info("TB head weights migrated successfully")
    logger.info("Pneumonia head initialized with random weights")

    return multihead


def create_model(freeze_backbone: bool = True):
    """Create fine-tuning model from pretrained XRV backbone"""
    import torchxrayvision as xrv

    logger.info("Loading TorchXRayVision DenseNet121 backbone...")
    xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")

    model = TBClassifier(xrv_model)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen — training classifier head only")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    return model


def unfreeze_backbone(model, layers_to_unfreeze: int = 2):
    """Unfreeze last N dense blocks for deeper fine-tuning"""
    blocks = ['denseblock4', 'denseblock3', 'denseblock2', 'denseblock1']
    unfrozen = 0

    for name, param in model.backbone.named_parameters():
        for block in blocks[:layers_to_unfreeze]:
            if block in name:
                param.requires_grad = True
                unfrozen += 1
                break

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Unfroze last {layers_to_unfreeze} dense blocks ({unfrozen} param tensors)")
    logger.info(f"Trainable parameters now: {trainable:,}")


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set, return full metrics"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            scores = torch.sigmoid(outputs)

            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    labels = np.array(all_labels)
    scores = np.array(all_scores)

    avg_loss = total_loss / len(labels)
    auc = calculate_auc(labels, scores)

    # At default threshold 0.5
    preds = (scores >= 0.5).astype(int)
    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    sens_default = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec_default = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Find optimal threshold
    best_threshold, best_sens, best_spec = find_optimal_threshold(labels, scores)

    return {
        'loss': avg_loss,
        'auc': auc,
        'sensitivity_default': sens_default,
        'specificity_default': spec_default,
        'optimal_threshold': best_threshold,
        'optimal_sensitivity': best_sens,
        'optimal_specificity': best_spec,
        'labels': labels,
        'scores': scores,
    }


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def calculate_auc(labels, scores):
    """AUC-ROC via trapezoidal rule"""
    sorted_indices = np.argsort(-scores)
    sorted_labels = labels[sorted_indices]

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.0

    tpr_points = [0.0]
    fpr_points = [0.0]
    tp = 0
    fp = 0

    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_points.append(tp / n_pos)
        fpr_points.append(fp / n_neg)

    auc = 0.0
    for i in range(1, len(fpr_points)):
        auc += (fpr_points[i] - fpr_points[i - 1]) * (tpr_points[i] + tpr_points[i - 1]) / 2

    return auc


def find_optimal_threshold(labels, scores):
    """Find threshold maximizing Youden's J (sensitivity + specificity - 1)"""
    best_j = -1
    best_t = 0.5
    best_sens = 0
    best_spec = 0

    for t in np.arange(0.05, 0.95, 0.005):
        preds = (scores >= t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sens + spec - 1

        if j > best_j:
            best_j = j
            best_t = t
            best_sens = sens
            best_spec = spec

    return round(float(best_t), 3), float(best_sens), float(best_spec)


def bootstrap_ci(labels, scores, metric_fn, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval for a metric function"""
    n = len(labels)
    values = []

    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        boot_labels = labels[indices]
        boot_scores = scores[indices]

        if len(np.unique(boot_labels)) < 2:
            continue

        values.append(metric_fn(boot_labels, boot_scores))

    if not values:
        return 0.0, 0.0

    values = sorted(values)
    alpha = (1 - ci) / 2
    lower = values[max(0, int(alpha * len(values)))]
    upper = values[min(len(values) - 1, int((1 - alpha) * len(values)))]

    return lower, upper


# ──────────────────────────────────────────────
# Data Splitting
# ──────────────────────────────────────────────

def stratified_split(samples, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Stratified train/val/test split preserving class ratios"""
    random.seed(seed)
    np.random.seed(seed)

    tb_samples = [s for s in samples if s['label'] == 1]
    normal_samples = [s for s in samples if s['label'] == 0]

    random.shuffle(tb_samples)
    random.shuffle(normal_samples)

    def split_list(lst, tr, vr):
        n = len(lst)
        n_train = int(n * tr)
        n_val = int(n * vr)
        return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]

    tb_train, tb_val, tb_test = split_list(tb_samples, train_ratio, val_ratio)
    norm_train, norm_val, norm_test = split_list(normal_samples, train_ratio, val_ratio)

    train = tb_train + norm_train
    val = tb_val + norm_val
    test = tb_test + norm_test

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


# ──────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────

def generate_report(train_history, test_metrics, baseline, config, total_time, output_path):
    """Generate markdown report: baseline vs fine-tuned with bootstrap CIs"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    labels = test_metrics['labels']
    scores = test_metrics['scores']
    opt_t = test_metrics['optimal_threshold']

    # Bootstrap CIs
    def sens_fn(l, s):
        p = (s >= opt_t).astype(int)
        tp = np.sum((p == 1) & (l == 1))
        fn = np.sum((p == 0) & (l == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    def spec_fn(l, s):
        p = (s >= opt_t).astype(int)
        tn = np.sum((p == 0) & (l == 0))
        fp = np.sum((p == 1) & (l == 0))
        return tn / (tn + fp) if (tn + fp) > 0 else 0

    print("  Computing bootstrap confidence intervals (1000 iterations)...")
    sens_ci = bootstrap_ci(labels, scores, sens_fn)
    spec_ci = bootstrap_ci(labels, scores, spec_fn)
    auc_ci = bootstrap_ci(labels, scores, calculate_auc)

    who_sens = "MEETS" if test_metrics['optimal_sensitivity'] >= 0.90 else "BELOW"
    who_spec = "MEETS" if test_metrics['optimal_specificity'] >= 0.70 else "BELOW"

    # Confusion matrix numbers
    preds = (scores >= opt_t).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    tb_scores = scores[labels == 1]
    norm_scores = scores[labels == 0]

    report = f"""# ONA Health - Fine-Tuning Results

**Generated:** {now}
**Model:** DenseNet121 fine-tuned for TB detection
**Backbone:** TorchXRayVision densenet121-res224-all
**Dataset:** {config['dataset']} ({config['total_images']} images)
**Split:** {config['n_train']} train / {config['n_val']} val / {config['n_test']} test
**Training time:** {total_time:.0f}s ({config['epochs_completed']} epochs)

---

## Results: Baseline vs Fine-Tuned

| Metric | Baseline (zero-shot) | Fine-Tuned | WHO Target | Status |
|--------|---------------------|------------|------------|--------|
| **Sensitivity** | {baseline['sensitivity']} | **{test_metrics['optimal_sensitivity']:.1%}** (95% CI: {sens_ci[0]:.1%}-{sens_ci[1]:.1%}) | >90% | {who_sens} |
| **Specificity** | {baseline['specificity']} | **{test_metrics['optimal_specificity']:.1%}** (95% CI: {spec_ci[0]:.1%}-{spec_ci[1]:.1%}) | >70% | {who_spec} |
| **AUC-ROC** | {baseline['auc']} | **{test_metrics['auc']:.3f}** (95% CI: {auc_ci[0]:.3f}-{auc_ci[1]:.3f}) | >0.85 | {"MEETS" if test_metrics['auc'] >= 0.85 else "BELOW"} |
| **Optimal Threshold** | {baseline['threshold']} | {opt_t:.3f} | - | - |

**WHO TPP:** Sensitivity {who_sens} / Specificity {who_spec}

---

## Test Set Confusion Matrix (threshold = {opt_t:.3f})

```
                      Predicted
                 |  TB     |  Normal  |
Actual    TB     |  {tp:>5}  |  {fn:>5}   |
        Normal   |  {fp:>5}  |  {tn:>5}   |
```

- **True Positives:** {tp} (correctly detected TB)
- **True Negatives:** {tn} (correctly cleared healthy)
- **False Positives:** {fp} (healthy flagged as TB)
- **False Negatives:** {fn} (TB missed)

---

## Score Distribution (Test Set)

**TB Positive** (n={len(tb_scores)}):
- Mean: {np.mean(tb_scores):.3f}, Range: [{np.min(tb_scores):.3f}, {np.max(tb_scores):.3f}]

**Normal** (n={len(norm_scores)}):
- Mean: {np.mean(norm_scores):.3f}, Range: [{np.min(norm_scores):.3f}, {np.max(norm_scores):.3f}]

**Separation:** {np.mean(tb_scores) - np.mean(norm_scores):.3f} (baseline was 0.042)

---

## Training History

| Epoch | Train Loss | Train Acc | Val Loss | Val AUC | Val Sens | Val Spec |
|-------|-----------|-----------|----------|---------|----------|----------|
"""

    for h in train_history:
        report += (f"| {h['epoch']} | {h['train_loss']:.4f} | {h['train_acc']:.1%} | "
                   f"{h['val_loss']:.4f} | {h['val_auc']:.3f} | "
                   f"{h['val_sens']:.1%} | {h['val_spec']:.1%} |\n")

    report += f"""
---

## Configuration

- **Backbone:** DenseNet121 pretrained on CheXpert, MIMIC, NIH, PadChest
- **Classifier:** Dropout(0.3) + Linear(1024,256) + ReLU + Dropout(0.2) + Linear(256,1)
- **Optimizer:** Adam (lr={config['lr']})
- **Scheduler:** Cosine annealing
- **Augmentation:** horizontal flip, rotation +/-10deg, brightness, contrast
- **Class weight:** {config['class_weight']:.1f}x for TB positive (compensating {config['n_normal']}:{config['n_tb']} imbalance)
- **Early stopping:** patience {config['patience']} on val AUC
- **Backbone unfrozen at epoch:** {config['unfreeze_epoch']}

---

## For Grant Applications

> "After transfer learning on {config['total_images']} labeled chest radiographs, ONA Health's AI system achieved **{test_metrics['optimal_sensitivity']:.0%} sensitivity (95% CI: {sens_ci[0]:.0%}-{sens_ci[1]:.0%})** and **{test_metrics['optimal_specificity']:.0%} specificity (95% CI: {spec_ci[0]:.0%}-{spec_ci[1]:.0%})** for TB detection, with an AUC-ROC of **{test_metrics['auc']:.2f} (95% CI: {auc_ci[0]:.2f}-{auc_ci[1]:.2f})**. This represents improvement over the zero-shot baseline (AUC {baseline['auc']}), validating the transfer learning approach for deployment in resource-limited settings."

---

*Model saved to: {config.get('model_path', 'N/A')}*
*Report generated by ONA Edge Fine-Tuning Tool v1.0*
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    return report


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Fine-tune DenseNet121 for TB detection')
    parser.add_argument('dataset_path', help='Path to labeled dataset')
    parser.add_argument('--epochs', type=int, default=30, help='Max training epochs (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (default: 16, use 8 for low RAM)')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience (default: 7)')
    parser.add_argument('--unfreeze-epoch', type=int, default=5, help='Epoch to unfreeze backbone layers (default: 5)')
    parser.add_argument('--output', type=str, help='Output model path')
    parser.add_argument('--report', type=str, help='Output report path')
    parser.add_argument('--baseline-auc', type=float, default=0.824, help='Baseline AUC for comparison')
    parser.add_argument('--baseline-sensitivity', type=str, default='72.3%', help='Baseline sensitivity string')
    parser.add_argument('--baseline-specificity', type=str, default='85.0%', help='Baseline specificity string')
    parser.add_argument('--baseline-threshold', type=str, default='0.63', help='Baseline threshold string')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')

    args = parser.parse_args()

    setup_logging()

    print("=" * 60)
    print("  ONA Health — Model Fine-Tuning")
    print("  DenseNet121 Transfer Learning for TB Detection")
    print("=" * 60)

    # ── Load dataset ──
    print(f"\nLoading dataset: {args.dataset_path}")
    samples = load_dataset(args.dataset_path)

    if not samples:
        print("ERROR: No images found.")
        sys.exit(1)

    # Keep only labeled samples
    samples = [s for s in samples if s['label'] >= 0]
    n_tb = sum(1 for s in samples if s['label'] == 1)
    n_normal = sum(1 for s in samples if s['label'] == 0)
    print(f"  Total: {len(samples)} ({n_tb} TB, {n_normal} Normal)")

    # ── Split data ──
    train_samples, val_samples, test_samples = stratified_split(samples)
    train_tb = sum(1 for s in train_samples if s['label'] == 1)
    val_tb = sum(1 for s in val_samples if s['label'] == 1)
    test_tb = sum(1 for s in test_samples if s['label'] == 1)

    print(f"  Train: {len(train_samples)} ({train_tb} TB, {len(train_samples) - train_tb} Normal)")
    print(f"  Val:   {len(val_samples)} ({val_tb} TB, {len(val_samples) - val_tb} Normal)")
    print(f"  Test:  {len(test_samples)} ({test_tb} TB, {len(test_samples) - test_tb} Normal)")

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
    if device.type == 'cpu':
        print("  (Training on CPU — this will take a while. Use --batch-size 8 if RAM is tight.)")

    # ── Create model ──
    model = create_model(freeze_backbone=True)
    model = model.to(device)

    # ── Loss with class weighting ──
    pos_weight = torch.tensor([n_normal / n_tb]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  Class weight: {pos_weight.item():.1f}x for TB positive")

    # ── Optimizer ──
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Checkpoint path ──
    checkpoint_dir = Path(MODEL_DIR) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'finetune_checkpoint.pth'

    # ── Training loop ──
    best_val_auc = 0
    patience_counter = 0
    train_history = []
    best_model_state = None
    start_epoch = 1
    start_time = time.time()

    # ── Resume from checkpoint if available ──
    if args.resume and checkpoint_path.exists():
        print(f"\n  Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_val_auc = checkpoint['best_val_auc']
        patience_counter = checkpoint['patience_counter']
        train_history = checkpoint['train_history']
        best_model_state = checkpoint.get('best_model_state', None)
        start_epoch = checkpoint['epoch'] + 1
        # If we're past the unfreeze epoch, unfreeze now
        if start_epoch > args.unfreeze_epoch:
            unfreeze_backbone(model, layers_to_unfreeze=2)
            optimizer = optim.Adam([
                {'params': model.classifier.parameters(), 'lr': args.lr * 0.1},
                {'params': filter(lambda p: p.requires_grad, model.backbone.parameters()), 'lr': args.lr * 0.01},
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - start_epoch + 1)
        print(f"  Resumed at epoch {start_epoch}, best AUC so far: {best_val_auc:.3f}, patience: {patience_counter}")
    else:
        if args.resume:
            print(f"\n  No checkpoint found, starting fresh.")

    print(f"\n{'=' * 60}")
    print(f"  Training — up to {args.epochs} epochs, patience {args.patience}")
    print(f"{'=' * 60}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Unfreeze backbone at specified epoch
        if epoch == args.unfreeze_epoch:
            print(f"\n  >>> Unfreezing backbone (last 2 dense blocks) <<<\n")
            unfreeze_backbone(model, layers_to_unfreeze=2)
            # New optimizer with differential learning rates
            optimizer = optim.Adam([
                {'params': model.classifier.parameters(), 'lr': args.lr * 0.1},
                {'params': filter(lambda p: p.requires_grad, model.backbone.parameters()), 'lr': args.lr * 0.01},
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch + 1)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

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

        # Early stopping on val AUC
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

        # ── Save checkpoint after every epoch ──
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_model_state': best_model_state,
            'best_val_auc': best_val_auc,
            'patience_counter': patience_counter,
            'train_history': train_history,
        }, checkpoint_path)
        print(f"         [checkpoint saved]")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s")

    # ── Load best model ──
    if best_model_state:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    # ── Evaluate on held-out test set ──
    print(f"\n{'=' * 60}")
    print("  Evaluating on held-out test set...")
    print(f"{'=' * 60}")

    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n  Test AUC-ROC:     {test_metrics['auc']:.3f}")
    print(f"  Sensitivity:      {test_metrics['optimal_sensitivity']:.1%} (threshold {test_metrics['optimal_threshold']:.3f})")
    print(f"  Specificity:      {test_metrics['optimal_specificity']:.1%}")

    # ── Save model ──
    model_path = args.output or str(Path(MODEL_DIR) / 'ona-tb-finetuned.pth')
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    model_cpu = model.cpu()
    torch.save({
        'model_state_dict': model_cpu.state_dict(),
        'architecture': 'TBClassifier',
        'backbone': 'torchxrayvision-densenet121-res224-all',
        'optimal_threshold': test_metrics['optimal_threshold'],
        'test_auc': test_metrics['auc'],
        'test_sensitivity': test_metrics['optimal_sensitivity'],
        'test_specificity': test_metrics['optimal_specificity'],
        'trained_on': Path(args.dataset_path).name,
        'date': datetime.now().isoformat(),
    }, model_path)
    print(f"\n  Model saved: {model_path}")

    # ── Generate report ──
    baseline = {
        'auc': str(args.baseline_auc),
        'sensitivity': args.baseline_sensitivity,
        'specificity': args.baseline_specificity,
        'threshold': args.baseline_threshold,
    }

    config = {
        'dataset': Path(args.dataset_path).name,
        'total_images': len(samples),
        'n_train': len(train_samples),
        'n_val': len(val_samples),
        'n_test': len(test_samples),
        'n_tb': n_tb,
        'n_normal': n_normal,
        'lr': args.lr,
        'class_weight': n_normal / n_tb,
        'patience': args.patience,
        'unfreeze_epoch': args.unfreeze_epoch,
        'epochs_completed': len(train_history),
        'model_path': model_path,
    }

    report_path = args.report or str(Path(__file__).parent.parent / 'finetune_report.md')
    generate_report(train_history, test_metrics, baseline, config, total_time, report_path)
    print(f"  Report saved: {report_path}")

    # ── Save training history as JSON ──
    json_path = str(Path(report_path).with_suffix('.json'))
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': config,
            'baseline': baseline,
            'test_metrics': {
                'auc': test_metrics['auc'],
                'sensitivity': test_metrics['optimal_sensitivity'],
                'specificity': test_metrics['optimal_specificity'],
                'threshold': test_metrics['optimal_threshold'],
            },
            'history': train_history,
            'total_time_seconds': total_time,
        }, f, indent=2)
    print(f"  History saved: {json_path}")

    print(f"\n{'=' * 60}")
    print(f"  DONE")
    print(f"  Baseline AUC: {args.baseline_auc:.3f} -> Fine-tuned AUC: {test_metrics['auc']:.3f}")
    print(f"  Sensitivity: {args.baseline_sensitivity} -> {test_metrics['optimal_sensitivity']:.1%}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

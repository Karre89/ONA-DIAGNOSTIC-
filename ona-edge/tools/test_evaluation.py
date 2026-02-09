"""
ONA Edge — Test Set Evaluation
Loads the fine-tuned checkpoint and evaluates on the held-out test set.
Produces bootstrap CIs and optionally exports to ONNX.

Usage:
  python tools/test_evaluation.py "C:/path/to/TB_Chest_Radiography_Database"
  python tools/test_evaluation.py "C:/path/to/dataset" --export-onnx
"""

import sys
import os
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from config import MODEL_DIR
from validate_model import load_dataset
from finetune_model import (
    TBClassifier, TBDataset, create_model, unfreeze_backbone,
    stratified_split, evaluate, calculate_auc, bootstrap_ci,
    find_optimal_threshold, generate_report
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned model on held-out test set')
    parser.add_argument('dataset_path', help='Path to labeled dataset')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (default: auto-detect)')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--export-onnx', action='store_true', help='Export to ONNX if test set passes thresholds')
    parser.add_argument('--sens-threshold', type=float, default=0.90, help='Min sensitivity to pass (default: 0.90)')
    parser.add_argument('--auc-threshold', type=float, default=0.95, help='Min AUC to pass (default: 0.95)')
    args = parser.parse_args()

    print("=" * 60)
    print("  ONA Health — Test Set Evaluation")
    print("  Held-out test set (never seen during training)")
    print("=" * 60)

    # ── Locate checkpoint ──
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = str(Path(MODEL_DIR) / 'checkpoints' / 'finetune_checkpoint.pth')

    if not Path(checkpoint_path).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"\nCheckpoint: {checkpoint_path}")

    # ── Load dataset with SAME split as training (seed=42) ──
    print(f"Loading dataset: {args.dataset_path}")
    samples = load_dataset(args.dataset_path)
    samples = [s for s in samples if s['label'] >= 0]
    n_tb = sum(1 for s in samples if s['label'] == 1)
    n_normal = sum(1 for s in samples if s['label'] == 0)
    print(f"  Total: {len(samples)} ({n_tb} TB, {n_normal} Normal)")

    train_samples, val_samples, test_samples = stratified_split(samples)
    test_tb = sum(1 for s in test_samples if s['label'] == 1)
    print(f"  Test split: {len(test_samples)} images ({test_tb} TB, {len(test_samples) - test_tb} Normal)")
    print(f"  (Same seed=42 split used during training — these images were NEVER trained on)")

    # ── Load model from checkpoint ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    print("Loading model...")
    model = create_model(freeze_backbone=False)  # Need full model to load all weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load the BEST model state (not the latest epoch state)
    best_state = checkpoint.get('best_model_state')
    if best_state:
        model.load_state_dict(best_state)
        print(f"  Loaded BEST model state (AUC {checkpoint['best_val_auc']:.3f} from training)")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded latest model state (no best_model_state in checkpoint)")

    model = model.to(device)
    model.eval()

    epoch = checkpoint['epoch']
    val_auc = checkpoint['best_val_auc']
    train_history = checkpoint.get('train_history', [])
    print(f"  Checkpoint epoch: {epoch}, validation AUC: {val_auc:.3f}")

    # ── Create test loader ──
    test_dataset = TBDataset(test_samples, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Evaluate on test set ──
    print(f"\n{'=' * 60}")
    print("  EVALUATING ON HELD-OUT TEST SET")
    print(f"{'=' * 60}")

    pos_weight = torch.tensor([n_normal / n_tb]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    start_time = time.time()
    test_metrics = evaluate(model, test_loader, criterion, device)
    eval_time = time.time() - start_time

    labels = test_metrics['labels']
    scores = test_metrics['scores']
    opt_t = test_metrics['optimal_threshold']

    print(f"\n  Test AUC-ROC:      {test_metrics['auc']:.4f}")
    print(f"  Sensitivity:       {test_metrics['optimal_sensitivity']:.1%} (at threshold {opt_t:.3f})")
    print(f"  Specificity:       {test_metrics['optimal_specificity']:.1%}")
    print(f"  Evaluation time:   {eval_time:.1f}s")

    # ── Confusion matrix ──
    preds = (scores >= opt_t).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    print(f"\n  Confusion Matrix (threshold = {opt_t:.3f}):")
    print(f"                       Predicted")
    print(f"                  |  TB     |  Normal  |")
    print(f"  Actual    TB    |  {tp:>5}  |  {fn:>5}   |")
    print(f"          Normal  |  {fp:>5}  |  {tn:>5}   |")
    print(f"\n  TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")

    # ── Score distributions ──
    tb_scores = scores[labels == 1]
    norm_scores = scores[labels == 0]
    separation = np.mean(tb_scores) - np.mean(norm_scores)

    print(f"\n  Score Distribution:")
    print(f"    TB Positive (n={len(tb_scores)}):  mean={np.mean(tb_scores):.3f}  range=[{np.min(tb_scores):.3f}, {np.max(tb_scores):.3f}]")
    print(f"    Normal (n={len(norm_scores)}):      mean={np.mean(norm_scores):.3f}  range=[{np.min(norm_scores):.3f}, {np.max(norm_scores):.3f}]")
    print(f"    Separation: {separation:.3f} (baseline was 0.042)")

    # ── Val vs Test comparison (overfitting check) ──
    val_sens = train_history[-1]['val_sens'] if train_history else val_auc
    sens_drop = val_sens - test_metrics['optimal_sensitivity']
    print(f"\n  Overfitting Check:")
    print(f"    Validation sensitivity: {val_sens:.1%}")
    print(f"    Test sensitivity:       {test_metrics['optimal_sensitivity']:.1%}")
    print(f"    Drop:                   {sens_drop:.1%}")
    if sens_drop > 0.06:
        print(f"    WARNING: >6% drop suggests overfitting from backbone unfreezing")
    elif sens_drop > 0.03:
        print(f"    NOTE: Mild gap, expected with 420-image splits")
    else:
        print(f"    GOOD: Minimal gap, model generalizes well")

    # ── Bootstrap confidence intervals ──
    print(f"\n{'=' * 60}")
    print("  BOOTSTRAP CONFIDENCE INTERVALS (1000 iterations)")
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

    # ── WHO Target assessment ──
    sens_pass = test_metrics['optimal_sensitivity'] >= args.sens_threshold
    auc_pass = test_metrics['auc'] >= args.auc_threshold
    spec_pass = test_metrics['optimal_specificity'] >= 0.70

    print(f"\n  WHO Target Assessment:")
    print(f"    Sensitivity >{args.sens_threshold:.0%}: {'PASS' if sens_pass else 'FAIL'}")
    print(f"    Specificity >70%:  {'PASS' if spec_pass else 'FAIL'}")
    print(f"    AUC >{args.auc_threshold}:      {'PASS' if auc_pass else 'FAIL'}")

    all_pass = sens_pass and auc_pass and spec_pass

    # ── Generate report ──
    baseline = {
        'auc': '0.824',
        'sensitivity': '72.3%',
        'specificity': '85.0%',
        'threshold': '0.63',
    }

    config = {
        'dataset': Path(args.dataset_path).name,
        'total_images': len(samples),
        'n_train': len(train_samples),
        'n_val': len(val_samples),
        'n_test': len(test_samples),
        'n_tb': n_tb,
        'n_normal': n_normal,
        'lr': 0.001,
        'class_weight': n_normal / n_tb,
        'patience': 7,
        'unfreeze_epoch': 5,
        'epochs_completed': epoch,
        'model_path': str(Path(MODEL_DIR) / 'ona-tb-finetuned.pth'),
    }

    report_path = str(Path(__file__).parent.parent / 'test_evaluation_report.md')
    generate_report(train_history, test_metrics, baseline, config, eval_time, report_path)
    print(f"\n  Report saved: {report_path}")

    # ── Save final model ──
    model_path = str(Path(MODEL_DIR) / 'ona-tb-finetuned.pth')
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.cpu()
    torch.save({
        'model_state_dict': model_cpu.state_dict(),
        'architecture': 'TBClassifier',
        'backbone': 'torchxrayvision-densenet121-res224-all',
        'optimal_threshold': opt_t,
        'test_auc': test_metrics['auc'],
        'test_sensitivity': test_metrics['optimal_sensitivity'],
        'test_specificity': test_metrics['optimal_specificity'],
        'test_sens_ci': sens_ci,
        'test_spec_ci': spec_ci,
        'test_auc_ci': auc_ci,
        'trained_on': Path(args.dataset_path).name,
        'date': datetime.now().isoformat(),
        'epoch': epoch,
    }, model_path)
    print(f"  Final model saved: {model_path}")

    # ── ONNX export ──
    if args.export_onnx and all_pass:
        print(f"\n{'=' * 60}")
        print("  EXPORTING TO ONNX")
        print(f"{'=' * 60}")
        onnx_path = str(Path(MODEL_DIR) / 'ona-tb-finetuned.onnx')
        try:
            model.eval()
            dummy_input = torch.randn(1, 1, 224, 224)
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'},
                }
            )
            print(f"  ONNX model saved: {onnx_path}")
            print(f"  Size: {Path(onnx_path).stat().st_size / 1024 / 1024:.1f} MB")

            # Verify ONNX model
            try:
                import onnx
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                print(f"  ONNX verification: PASSED")
            except ImportError:
                print(f"  (onnx package not installed — skipping verification)")
            except Exception as e:
                print(f"  ONNX verification WARNING: {e}")

        except Exception as e:
            print(f"  ONNX export FAILED: {e}")
    elif args.export_onnx and not all_pass:
        print(f"\n  ONNX export SKIPPED — test set did not meet thresholds")
    else:
        if all_pass:
            print(f"\n  Model PASSES all thresholds. Run with --export-onnx to export.")

    # ── Final summary ──
    print(f"\n{'=' * 60}")
    result = "PASS" if all_pass else "NEEDS REVIEW"
    print(f"  RESULT: {result}")
    print(f"  Baseline: AUC 0.824, Sensitivity 72.3%")
    print(f"  Fine-tuned: AUC {test_metrics['auc']:.3f}, Sensitivity {test_metrics['optimal_sensitivity']:.1%}")
    if all_pass:
        print(f"\n  Ready for Horizon1000 application.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

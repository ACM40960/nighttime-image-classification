import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from dataset import build_datasets, get_dataloaders
from train import Dinov2Classifier

# Inference
@torch.no_grad()
def get_predictions(model, loader, device, num_classes):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = F.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)
        all_labels.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.append(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_probs),
    )

# Metric helpers
def per_class_binary_counts(y_true, y_pred, num_classes):
    """
    For each class c, treat it as a one-vs-rest binary problem and count: TP, TN, FP, FN
    Returns four arrays of shape (num_classes,).
    """
    tp = np.zeros(num_classes, dtype=int)
    tn = np.zeros(num_classes, dtype=int)
    fp = np.zeros(num_classes, dtype=int)
    fn = np.zeros(num_classes, dtype=int)

    for c in range(num_classes):
        binary_true = (y_true == c).astype(int)
        binary_pred = (y_pred == c).astype(int)
        tp[c] = ((binary_pred == 1) & (binary_true == 1)).sum()
        tn[c] = ((binary_pred == 0) & (binary_true == 0)).sum()
        fp[c] = ((binary_pred == 1) & (binary_true == 0)).sum()
        fn[c] = ((binary_pred == 0) & (binary_true == 1)).sum()

    return tp, tn, fp, fn

def safe_divide(num, denom):
    """Element-wise division, returning 0 where denom == 0."""
    return np.where(denom > 0, num / denom, 0.0)

def compute_roc_auc(y_true_binary, y_score):
    """
    Compute AUC-ROC for a single binary problem.
    """
    # Sort by descending score
    order   = np.argsort(-y_score)
    y_true_ = y_true_binary[order]

    n_pos = y_true_.sum()
    n_neg = len(y_true_) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp_arr, fp_arr = [], []
    tp = fp = 0
    for label in y_true_:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tp_arr.append(tp)
        fp_arr.append(fp)

    tpr = np.array(tp_arr) / n_pos
    fpr = np.array(fp_arr) / n_neg

    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    auc = float(np.trapezoid(tpr, fpr))
    return auc


def build_confusion_matrix(y_true, y_pred, num_classes):
    """Return a (num_classes, num_classes) confusion matrix.
    Row = true class, column = predicted class."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

# Metrics computation
def compute_all_metrics(y_true, y_pred, y_probs, num_classes, idx2label):
    tp, tn, fp, fn = per_class_binary_counts(y_true, y_pred, num_classes)

    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    f1 = safe_divide(2 * tp, 2 * tp + fp + fn)
    # Accuracy for class c in one-vs-rest sense
    acc_per_cls = safe_divide(tp + tn,  tp + tn + fp + fn)

    auc_per_cls = np.array([
        compute_roc_auc((y_true == c).astype(int), y_probs[:, c])
        for c in range(num_classes)
    ])

    per_class = []
    for c in range(num_classes):
        per_class.append({
            "class":       idx2label[c],
            "n_samples":   int((y_true == c).sum()),
            "TP":          int(tp[c]),
            "TN":          int(tn[c]),
            "FP":          int(fp[c]),
            "FN":          int(fn[c]),
            "sensitivity": float(sensitivity[c]),
            "specificity": float(specificity[c]),
            "precision":   float(precision[c]),
            "accuracy":    float(acc_per_cls[c]),
            "f1":          float(f1[c]),
            "auc_roc":     float(auc_per_cls[c]),
        })

    # Macro averages (mean over classes that have test samples)
    has_samples = np.array([(y_true == c).sum() > 0 for c in range(num_classes)])

    def macro(arr):
        vals = arr[has_samples & ~np.isnan(arr)]
        return float(vals.mean()) if len(vals) > 0 else float("nan")

    # Overall accuracy is a single number
    overall_acc = float((y_true == y_pred).sum() / len(y_true))

    overall = {
        "accuracy": overall_acc,
        "sensitivity": macro(sensitivity),
        "specificity": macro(specificity),
        "precision": macro(precision),
        "f1": macro(f1),
        "auc_roc": macro(auc_per_cls),
    }

    return {"per_class": per_class, "overall": overall}

# Reporting
def save_confusion_matrix(cm, class_names, out_path):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 2)))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title("Confusion Matrix — voc_night (test)", fontsize=13)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix figure → {out_path}")


def save_text_report(metrics, out_path):
    ov = metrics["overall"]
    lines = [
        "=" * 65,
        "  DINOv2-base  |  Wildlife Camera Trap  |  Night-Time Test",
        "=" * 65,
        "",
        "OVERALL (macro-averaged)",
        "-" * 40,
        f"  Accuracy     : {ov['accuracy']:.4f}",
        f"  Sensitivity  : {ov['sensitivity']:.4f}  (macro recall)",
        f"  Specificity  : {ov['specificity']:.4f}",
        f"  Precision    : {ov['precision']:.4f}  (macro)",
        f"  F1-score     : {ov['f1']:.4f}  (macro)",
        f"  AUC-ROC      : {ov['auc_roc']:.4f}  (macro, one-vs-rest)",
        "",
        "PER-SPECIES",
        "-" * 40,
    ]

    header = (f"{'Class':<22} {'N':>5} {'Sens':>6} {'Spec':>6} "
              f"{'Prec':>6} {'Acc':>6} {'F1':>6} {'AUC':>6}")
    lines.append(header)
    lines.append("-" * len(header))

    for pc in metrics["per_class"]:
        auc_str = f"{pc['auc_roc']:.4f}" if not np.isnan(pc["auc_roc"]) else "  N/A "
        lines.append(
            f"{pc['class']:<22} {pc['n_samples']:>5} "
            f"{pc['sensitivity']:>6.4f} {pc['specificity']:>6.4f} "
            f"{pc['precision']:>6.4f} {pc['accuracy']:>6.4f} "
            f"{pc['f1']:>6.4f} {auc_str:>6}"
        )

    lines += ["", "=" * 65]
    report = "\n".join(lines)
    print("\n" + report)
    out_path.write_text(report)
    print(f"\nReport saved → {out_path}")


def save_per_class_csv(metrics, out_path):
    import csv
    fields = ["class", "n_samples", "TP", "TN", "FP", "FN",
              "sensitivity", "specificity", "precision", "accuracy", "f1", "auc_roc"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for pc in metrics["per_class"]:
            w.writerow({k: (f"{pc[k]:.6f}" if isinstance(pc[k], float) else pc[k])
                        for k in fields})
    print(f"Per-class CSV  → {out_path}")

# Main
def evaluate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train.py first to generate outputs/best_model.pt"
        )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    label2idx  = ckpt["label2idx"]
    idx2label  = {int(k): v for k, v in ckpt["idx2label"].items()}
    num_classes = ckpt["num_classes"]

    print(f"Checkpoint: epoch {ckpt['epoch']}  |  {num_classes} classes")

    model = Dinov2Classifier(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Dataset
    _, test_ds, _, _ = build_datasets(args.data_root)
    _, test_loader   = get_dataloaders(
        test_ds, test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Inference
    print("\nRunning inference on voc_night …")
    y_true, y_pred, y_probs = get_predictions(
        model, test_loader, device, num_classes
    )
    print(f"Samples evaluated: {len(y_true)}")

    # Metrics
    metrics = compute_all_metrics(y_true, y_pred, y_probs, num_classes, idx2label)

    # Save outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = [idx2label[c] for c in range(num_classes)]

    save_text_report(metrics, out_dir / "evaluation_report.txt")
    save_per_class_csv(metrics, out_dir / "metrics_per_class.csv")

    cm = build_confusion_matrix(y_true, y_pred, num_classes)
    save_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")

    # Raw confusion matrix CSV
    import csv
    with open(out_dir / "confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ pred"] + class_names)
        for i, row in enumerate(cm):
            w.writerow([class_names[i]] + row.tolist())
    print(f"Confusion CSV  → {out_dir / 'confusion_matrix.csv'}")

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained DINOv2 on voc_night.")
    p.add_argument("--checkpoint",   default="outputs/best_model.pt")
    p.add_argument("--data_root",    default="data")
    p.add_argument("--output_dir",   default="outputs")
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--num_workers",  type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
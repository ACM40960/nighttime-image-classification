import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import build_datasets, get_dataloaders
from train import Dinov2Classifier

# Inference
@torch.no_grad()
def get_predictions(model, loader, device):
    """
    Run the model over `loader` and collect:
      all_labels : (N,)     true integer class indices
      all_preds  : (N,)     argmax predictions
      all_probs  : (N, C)   softmax probability vectors
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        output = model(imgs)
        logits = output[0] if isinstance(output, tuple) else output
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
    """One-vs-rest TP, TN, FP, FN for each class."""
    tp = np.zeros(num_classes, dtype=int)
    tn = np.zeros(num_classes, dtype=int)
    fp = np.zeros(num_classes, dtype=int)
    fn = np.zeros(num_classes, dtype=int)

    for c in range(num_classes):
        bt = (y_true == c).astype(int)
        bp = (y_pred == c).astype(int)
        tp[c] = ((bp == 1) & (bt == 1)).sum()
        tn[c] = ((bp == 0) & (bt == 0)).sum()
        fp[c] = ((bp == 1) & (bt == 0)).sum()
        fn[c] = ((bp == 0) & (bt == 1)).sum()

    return tp, tn, fp, fn

def safe_divide(num, denom):
    return np.where(denom > 0, num / denom, 0.0)

def compute_roc_auc(y_true_binary, y_score):
    """AUC-ROC via trapezoidal rule for one binary problem."""
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

    tpr = np.concatenate([[0.0], np.array(tp_arr) / n_pos])
    fpr = np.concatenate([[0.0], np.array(fp_arr) / n_neg])
    return float(np.trapezoid(tpr, fpr))


def compute_pr_curve(y_true_binary, y_score):
    """
    Compute precision-recall curve for one binary problem.

    Returns (precision, recall) arrays starting at recall=0.
    Uses the step-function interpolation convention (no smoothing).
    """
    order   = np.argsort(-y_score)
    y_true_ = y_true_binary[order]

    n_pos = y_true_.sum()
    if n_pos == 0:
        return np.array([0.0, 0.0]), np.array([0.0, 1.0])

    tp_arr, fp_arr = [], []
    tp = fp = 0
    for label in y_true_:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tp_arr.append(tp)
        fp_arr.append(fp)

    tp_arr = np.array(tp_arr)
    fp_arr = np.array(fp_arr)

    precision = tp_arr / (tp_arr + fp_arr)
    recall    = tp_arr / n_pos

    # Prepend point at recall=0, precision=1 (standard convention)
    precision = np.concatenate([[1.0], precision])
    recall    = np.concatenate([[0.0], recall])
    return precision, recall


def compute_pr_auc(precision, recall):
    """Area under the PR curve via trapezoidal rule."""
    return float(np.trapezoid(precision, recall))

# Confusion matrix
def build_confusion_matrix(y_true, y_pred, num_classes):
    """Return a (num_classes, num_classes) row-normalised fraction matrix."""
    cm_counts = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm_counts[t][p] += 1

    # Row-normalise: each cell = fraction of true-class samples predicted as that class
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_frac  = np.where(row_sums > 0, cm_counts / row_sums, 0.0)
    return cm_counts, cm_frac

# Metrics computation
def compute_all_metrics(y_true, y_pred, y_probs, num_classes, idx2label):
    tp, tn, fp, fn = per_class_binary_counts(y_true, y_pred, num_classes)

    sensitivity = safe_divide(tp,      tp + fn)
    specificity = safe_divide(tn,      tn + fp)
    precision   = safe_divide(tp,      tp + fp)
    f1          = safe_divide(2 * tp,  2 * tp + fp + fn)
    acc_per_cls = safe_divide(tp + tn, tp + tn + fp + fn)

    auc_roc_per_cls = np.array([
        compute_roc_auc((y_true == c).astype(int), y_probs[:, c])
        for c in range(num_classes)
    ])

    pr_curves = []
    pr_auc_per_cls = np.zeros(num_classes)
    for c in range(num_classes):
        prec, rec = compute_pr_curve((y_true == c).astype(int), y_probs[:, c])
        pr_auc_per_cls[c] = compute_pr_auc(prec, rec)
        pr_curves.append((prec, rec))

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
            "auc_roc":     float(auc_roc_per_cls[c]),
            "pr_auc":      float(pr_auc_per_cls[c]),
        })

    has_samples = np.array([(y_true == c).sum() > 0 for c in range(num_classes)])

    def macro(arr):
        vals = arr[has_samples & ~np.isnan(arr)]
        return float(vals.mean()) if len(vals) > 0 else float("nan")

    overall = {
        "accuracy":    float((y_true == y_pred).sum() / len(y_true)),
        "sensitivity": macro(sensitivity),
        "specificity": macro(specificity),
        "precision":   macro(precision),
        "f1":          macro(f1),
        "auc_roc":     macro(auc_roc_per_cls),
        "pr_auc":      macro(pr_auc_per_cls),
    }

    return {"per_class": per_class, "overall": overall, "pr_curves": pr_curves}

# Reporting
def save_confusion_matrix(cm_counts, cm_frac, class_names, out_path):
    """
    Save row-normalised confusion matrix heatmap.

    Cell values are displayed as fractions (e.g. 0.900).
    Colour intensity reflects the fraction.
    """
    n   = len(class_names)
    fig, ax = plt.subplots(figsize=(max(16, n * 2), max(12, (n - 2) * 2)))

    im = ax.imshow(cm_frac, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=18)
    ax.set_yticklabels(class_names, fontsize=18)
    ax.set_xlabel("Predicted", fontsize=22)
    ax.set_ylabel("True",      fontsize=22)
    ax.set_title("Confusion Matrix - voc_night (row-normalised fractions)", fontsize=26)

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm_frac[i, j]:.3f}",
                    ha="center", va="center",
                    color="white" if cm_frac[i, j] > thresh else "black",
                    fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix figure saved: {out_path}")


def save_pr_auc_plot(metrics, class_names, out_path):
    """
    Save a PR curve plot — one curve per species plus the macro average.
    """
    pr_curves    = metrics["pr_curves"]
    per_class    = metrics["per_class"]
    macro_pr_auc = metrics["overall"]["pr_auc"]

    fig, ax = plt.subplots(figsize=(20, 14))

    cmap = plt.get_cmap("tab20")
    for c, (prec, rec) in enumerate(pr_curves):
        label = f"{class_names[c]} (AUC={per_class[c]['pr_auc']:.3f})"
        ax.plot(rec, prec, color=cmap(c / len(pr_curves)),
                linewidth=2.4, alpha=0.8, label=label)

    ax.set_xlabel("Recall",    fontsize=24)
    ax.set_ylabel("Precision", fontsize=24)
    ax.set_title(f"Precision-Recall Curves - voc_night\n"
                 f"Macro PR-AUC = {macro_pr_auc:.4f}", fontsize=26)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(loc="lower left", fontsize=14, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PR-AUC plot saved: {out_path}")


def save_text_report(metrics, out_path):
    ov = metrics["overall"]
    lines = [
        "=" * 72,
        "  DINOv2-base  |  Wildlife Camera Trap  |  Night-Time Test",
        "=" * 72,
        "",
        "OVERALL (macro-averaged)",
        "-" * 45,
        f"  Accuracy     : {ov['accuracy']:.4f}",
        f"  Sensitivity  : {ov['sensitivity']:.4f}",
        f"  Specificity  : {ov['specificity']:.4f}",
        f"  Precision    : {ov['precision']:.4f}",
        f"  F1-score     : {ov['f1']:.4f}",
        f"  AUC-ROC      : {ov['auc_roc']:.4f}  (macro, one-vs-rest)",
        f"  PR-AUC       : {ov['pr_auc']:.4f}  (macro, one-vs-rest)",
        "",
        "PER-SPECIES",
        "-" * 45,
    ]

    header = (f"{'Class':<22} {'N':>5} {'Sens':>6} {'Spec':>6} "
              f"{'Prec':>6} {'Acc':>6} {'F1':>6} {'ROC':>6} {'PR':>6}")
    lines.append(header)
    lines.append("-" * len(header))

    for pc in metrics["per_class"]:
        roc_str = f"{pc['auc_roc']:.4f}" if not np.isnan(pc["auc_roc"]) else "  N/A "
        pr_str  = f"{pc['pr_auc']:.4f}"
        lines.append(
            f"{pc['class']:<22} {pc['n_samples']:>5} "
            f"{pc['sensitivity']:>6.4f} {pc['specificity']:>6.4f} "
            f"{pc['precision']:>6.4f} {pc['accuracy']:>6.4f} "
            f"{pc['f1']:>6.4f} {roc_str:>6} {pr_str:>6}"
        )

    lines += ["", "=" * 72]
    report = "\n".join(lines)
    print("\n" + report)
    out_path.write_text(report)
    print(f"Report saved: {out_path}")


def save_per_class_csv(metrics, out_path):
    fields = ["class", "n_samples", "TP", "TN", "FP", "FN",
              "sensitivity", "specificity", "precision", "accuracy",
              "f1", "auc_roc", "pr_auc"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for pc in metrics["per_class"]:
            w.writerow({k: (f"{pc[k]:.6f}" if isinstance(pc[k], float) else pc[k])
                        for k in fields})
    print(f"Per-class CSV saved: {out_path}")


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
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt        = torch.load(checkpoint_path, map_location=device)
    label2idx   = ckpt["label2idx"]
    idx2label   = {int(k): v for k, v in ckpt["idx2label"].items()}
    num_classes = ckpt["num_classes"]
    run_ts      = getattr(args, "run_ts", "run")
    use_supcon  = ckpt.get("use_supcon", False)

    print(f"Checkpoint: epoch {ckpt['epoch']}, {num_classes} classes.")

    model = Dinov2Classifier(
        num_classes=num_classes, use_supcon=use_supcon
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Use the 80% test split of voc_night.
    _, test_ds, _, _, _ = build_datasets(
        args.data_root, use_data_adapt=False
    )
    _, test_loader, _ = get_dataloaders(
        test_ds, test_ds, test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("\nRunning inference on voc_night test split.")
    y_true, y_pred, y_probs = get_predictions(model, test_loader, device)
    print(f"Samples evaluated: {len(y_true)}")

    metrics = compute_all_metrics(y_true, y_pred, y_probs, num_classes, idx2label)

    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = [idx2label[c] for c in range(num_classes)]
    ts          = run_ts

    save_text_report(metrics,   out_dir / f"evaluation_report_{ts}.txt")
    save_per_class_csv(metrics, out_dir / f"metrics_per_class_{ts}.csv")
    save_pr_auc_plot(metrics, class_names, out_dir / f"pr_auc_{ts}.png")

    cm_counts, cm_frac = build_confusion_matrix(y_true, y_pred, num_classes)
    save_confusion_matrix(cm_counts, cm_frac, class_names,
                          out_dir / f"confusion_matrix_{ts}.png")

    with open(out_dir / f"confusion_matrix_{ts}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true vs pred"] + class_names)
        for i, row in enumerate(cm_frac):
            w.writerow([class_names[i]] + [f"{v:.6f}" for v in row])
    print(f"Confusion matrix CSV saved: {out_dir / f'confusion_matrix_{ts}.csv'}")


# Entry point
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained DINOv2 on voc_night.")
    p.add_argument("--checkpoint",  default="outputs/best_model.pt")
    p.add_argument("--data_root",   default="./data")
    p.add_argument("--output_dir",  default="./outputs")
    p.add_argument("--run_ts",      default="run")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
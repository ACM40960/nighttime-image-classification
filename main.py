import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from dataset import build_datasets, get_dataloaders
from train import train, Dinov2Classifier
import evaluate as eval_module

# Configurations
CONFIG = dict(
    data_root      = "data",
    output_dir     = "outputs",
    epochs         = 20,
    warmup_epochs  = 5,
    batch_size     = 16,
    lr             = 1e-4,
    num_workers    = 4,
)

def banner(title: str, width: int = 60):
    print(f"\n{'='*width}")
    pad = (width - len(title) - 2) // 2
    print(f"{'='*pad} {title} {'='*(width - pad - len(title) - 2)}")
    print(f"{'='*width}\n")

# Step 1 — Verify
def step_verify(cfg) -> tuple:
    """Scan both splits and return (train_ds, test_ds, label2idx, idx2label)."""
    banner("STEP 1 / 3  —  DATA VERIFICATION")

    train_ds, test_ds, label2idx, idx2label = build_datasets(cfg.data_root)

    if len(train_ds) == 0:
        print("[ERROR] Training set is empty. Check voc_day/Annotations and voc_day/JPEGImages.")
        sys.exit(1)
    if len(test_ds) == 0:
        print("[ERROR] Test set is empty. Check voc_night/Annotations and voc_night/JPEGImages.")
        sys.exit(1)

    print(f"  Train samples : {len(train_ds)}")
    print(f"  Test samples  : {len(test_ds)}")
    print(f"  Classes ({len(label2idx)}) : {sorted(label2idx)}")
    print("\n Data verification passed.")

    return train_ds, test_ds, label2idx, idx2label

# Step 2 — Train
def step_train(cfg):
    """Run the two-phase training loop and save the best checkpoint."""
    banner("STEP 2 / 3  —  TRAINING")

    t0 = time.time()
    train(cfg)
    elapsed = time.time() - t0

    checkpoint = Path(cfg.output_dir) / "best_model.pt"
    if not checkpoint.exists():
        print("[ERROR] Training finished but no checkpoint was saved.")
        sys.exit(1)

    mins, secs = divmod(int(elapsed), 60)
    print(f"\n Training complete in {mins}m {secs}s.")
    print(f"  Checkpoint : {checkpoint}")

# Step 3 — Evaluate
def step_evaluate(cfg):
    """Load the best checkpoint and run the full evaluation suite."""
    banner("STEP 3 / 3  —  EVALUATION")

    eval_args = SimpleNamespace(
        checkpoint  = str(Path(cfg.output_dir) / "best_model.pt"),
        data_root   = cfg.data_root,
        output_dir  = cfg.output_dir,
        batch_size  = cfg.batch_size,
        num_workers = cfg.num_workers,
    )

    eval_module.evaluate(eval_args)

    out = Path(cfg.output_dir)
    print(f"\n  Evaluation complete. Outputs in {out}/")
    print(f" {out}/evaluation_report.txt")
    print(f" {out}/metrics_per_class.csv")
    print(f" {out}/confusion_matrix.png")
    print(f" {out}/confusion_matrix.csv")

# Summary footer
def print_summary(cfg, total_elapsed: float):
    banner("PIPELINE COMPLETE", width=60)
    mins, secs = divmod(int(total_elapsed), 60)
    print(f"  Total wall time : {mins}m {secs}s")
    print(f"  Outputs         : {Path(cfg.output_dir).resolve()}")
    print()

# CLI
def parse_args():
    p = argparse.ArgumentParser(
        description="Wildlife camera-trap classifier — full pipeline (DINOv2-base).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_root",     default=CONFIG["data_root"],
                   help="Root directory containing voc_day/ and voc_night/")
    p.add_argument("--output_dir",    default=CONFIG["output_dir"],
                   help="Directory for checkpoints, logs, and evaluation outputs")
    p.add_argument("--epochs",        type=int,   default=CONFIG["epochs"])
    p.add_argument("--warmup_epochs", type=int,   default=CONFIG["warmup_epochs"],
                   help="Epochs to train the classification head with the backbone frozen")
    p.add_argument("--batch_size",    type=int,   default=CONFIG["batch_size"],
                   help="Reduce to 8 if GPU runs out of memory")
    p.add_argument("--lr",            type=float, default=CONFIG["lr"],
                   help="Head learning rate; backbone uses lr × 0.1 during fine-tune phase")
    p.add_argument("--num_workers",   type=int,   default=CONFIG["num_workers"],
                   help="DataLoader workers — use 0 on Windows or for debugging")
    p.add_argument("--eval_only",     action="store_true",
                   help="Skip training and run evaluation on an existing checkpoint")
    return p.parse_args()

# Entry point
def main():
    args = parse_args()
    cfg = args

    banner("WILDLIFE SPECIES CLASSIFIER  —  DINOv2-BASE BASELINE", width=60)
    print(f" data_root     : {cfg.data_root}")
    print(f" output_dir    : {cfg.output_dir}")
    if not cfg.eval_only:
        print(f"  epochs        : {cfg.epochs}  (warmup: {cfg.warmup_epochs})")
        print(f"  batch_size    : {cfg.batch_size}")
        print(f"  learning rate : {cfg.lr}  (backbone: {cfg.lr * 0.1})")
        print(f"  num_workers   : {cfg.num_workers}")
    else:
        print("  mode          : eval-only (skipping training)")

    t_start = time.time()

    # Step 1 — verify data
    step_verify(cfg)

    # Step 2 — train (unless --eval_only)
    if not cfg.eval_only:
        step_train(cfg)
    else:
        checkpoint = Path(cfg.output_dir) / "best_model.pt"
        if not checkpoint.exists():
            print(f"[ERROR] --eval_only requested but no checkpoint at {checkpoint}")
            print("Run without --eval_only first to train the model.")
            sys.exit(1)
        print(f"  Skipping training. Using checkpoint: {checkpoint}")

    # Step 3 — evaluate
    step_evaluate(cfg)

    print_summary(cfg, time.time() - t_start)

if __name__ == "__main__":
    main()
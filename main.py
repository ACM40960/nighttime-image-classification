import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from dataset import build_datasets
from train import train
import evaluate as eval_module


# Fixed output path and default data root
DATA_ROOT  = "./data"
OUTPUT_DIR = "./outputs"

# Default configuration
CONFIG = dict(
    epochs          = 30,
    warmup_epochs   = 15,
    finetune_blocks = 3,
    batch_size      = 76,
    lr              = 1e-4,
    num_workers     = 4,
    use_data_adapt  = False,
    use_supcon      = False,
)


# Utilities
def banner(title: str, width: int = 60):
    print(f"\n{'=' * width}")
    pad = (width - len(title) - 2) // 2
    print(f"{'=' * pad} {title} {'=' * (width - pad - len(title) - 2)}")
    print(f"{'=' * width}\n")


# Step 1 - Verify
def step_verify(cfg) -> tuple:
    """Scan both dataset splits and report sample counts."""
    banner("STEP 1 / 3  -  DATA VERIFICATION")

    train_ds, test_ds, val_night_ds, label2idx, idx2label = \
        build_datasets(cfg.data_root, use_data_adapt=cfg.use_data_adapt)

    if len(train_ds) == 0:
        print("Error: training set is empty. "
              "Check voc_day/Annotations and voc_day/JPEGImages.")
        sys.exit(1)
    if len(test_ds) == 0:
        print("Error: test set is empty. "
              "Check voc_night/Annotations and voc_night/JPEGImages.")
        sys.exit(1)

    print(f" Train      : {len(train_ds)}")
    print(f" Test       : {len(test_ds)}")
    print(f" Val night  : {len(val_night_ds)}")
    print(f" Classes ({len(label2idx)}) : {sorted(label2idx)}")
    print("\n  Data verification passed.")

    return train_ds, test_ds, val_night_ds, label2idx, idx2label


# Step 2 - Train
def step_train(cfg):
    """Run the two-phase training loop and save the best checkpoint."""
    banner("STEP 2 / 3  -  TRAINING")

    options = []
    if cfg.use_data_adapt:
        options.append("data adaptation")
    if cfg.use_supcon:
        options.append("supervised contrastive loss")
    if options:
        print(f" Active options: {', '.join(options)}")
    else:
        print(" Active options: none (baseline mode)")
    print()

    t0 = time.time()
    train(cfg)
    elapsed = time.time() - t0

    checkpoint = Path(cfg.output_dir) / f"best_model_{cfg.run_ts}.pt"
    if not checkpoint.exists():
        print("Error: training finished but no checkpoint was saved.")
        sys.exit(1)

    mins, secs = divmod(int(elapsed), 60)
    print(f"\n  Training complete in {mins}m {secs}s.")
    print(f" Checkpoint : {checkpoint}")


# Step 3 - Evaluate
def step_evaluate(cfg):
    """Load the best checkpoint and run the full evaluation suite."""
    banner("STEP 3 / 3  -  EVALUATION")

    eval_args = SimpleNamespace(
        checkpoint  = str(Path(cfg.output_dir) / f"best_model_{cfg.run_ts}.pt"),
        data_root   = cfg.data_root,
        output_dir  = cfg.output_dir,
        run_ts      = cfg.run_ts,
        batch_size  = cfg.batch_size,
        num_workers = cfg.num_workers,
    )

    eval_module.evaluate(eval_args)

    ts  = cfg.run_ts
    out = Path(cfg.output_dir)
    print(f"\n  Evaluation complete. Outputs written to {out}/")
    print(f" evaluation_report_{ts}.txt")
    print(f" metrics_per_class_{ts}.csv")
    print(f" confusion_matrix_{ts}.png")
    print(f" confusion_matrix_{ts}.csv")
    print(f" pr_auc_{ts}.png")


# Summary
def print_summary(cfg, total_elapsed: float):
    banner("PIPELINE COMPLETE", width=60)
    mins, secs = divmod(int(total_elapsed), 60)
    print(f" Run ID     : {cfg.run_ts}")
    print(f" Wall time  : {mins}m {secs}s")
    print(f" Outputs    : {Path(cfg.output_dir).resolve()}")
    print()

# CLI
def parse_args():
    p = argparse.ArgumentParser(
        description="Wildlife camera-trap classifier - DINOv2-base pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_root", default=DATA_ROOT,
                   help="Root directory containing voc_day/ and voc_night/.")
    p.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    p.add_argument("--warmup_epochs", type=int, default=CONFIG["warmup_epochs"],
                   help="Epochs to train with backbone fully frozen.")
    p.add_argument("--finetune_blocks", type=int, default=CONFIG["finetune_blocks"],
                   help="Number of trailing encoder blocks to unfreeze in Phase 2.")
    p.add_argument("--batch_size", type=int, default=CONFIG["batch_size"],
                   help="Training and evaluation batch size.")
    p.add_argument("--lr", type=float, default=CONFIG["lr"],
                   help="Head learning rate. Backbone uses lr * 0.05 in Phase 2.")
    p.add_argument("--num_workers", type=int, default=CONFIG["num_workers"],
                   help="DataLoader worker processes.")
    p.add_argument("--use_data_adapt", action="store_true",
                   default=CONFIG["use_data_adapt"],
                   help="Append night-simulated copies of daytime training images.")
    p.add_argument("--use_supcon", action="store_true",
                   default=CONFIG["use_supcon"],
                   help="Use supervised contrastive loss during the warmup phase.")
    return p.parse_args()

# Entry point
def main():
    args   = parse_args()
    run_ts = datetime.now().strftime("%d_%m_%y_%H_%M_%S")

    cfg = SimpleNamespace(
        **vars(args),
        output_dir = OUTPUT_DIR,
        run_ts     = run_ts,
    )

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    banner("WILDLIFE SPECIES CLASSIFIER  -  DINOv2-BASE", width=60)
    print(f" Run ID         : {cfg.run_ts}")
    print(f" Data root      : {cfg.data_root}")
    print(f" Output dir     : {cfg.output_dir}")
    print(f" Epochs         : {cfg.epochs} "
          f"(warmup: {cfg.warmup_epochs}, finetune blocks: {cfg.finetune_blocks})")
    print(f" Batch size     : {cfg.batch_size}")
    print(f" Learning rate  : {cfg.lr}")
    print(f" Workers        : {cfg.num_workers}")
    print(f" Data adapt     : {'on' if cfg.use_data_adapt else 'off'}")
    print(f" SupCon         : {'on' if cfg.use_supcon else 'off'}")

    t_start = time.time()

    step_verify(cfg)
    step_train(cfg)
    step_evaluate(cfg)

    print_summary(cfg, time.time() - t_start)


if __name__ == "__main__":
    main()
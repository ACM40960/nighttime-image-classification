import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import Dinov2Model

from dataset import build_datasets, get_dataloaders
from data_adaptation import get_adapted_train_transforms
from dann import DomainAdversarialModel, build_dann_datasets, get_dann_dataloaders, dann_train_one_epoch, compute_lambda

# Model
class Dinov2Classifier(nn.Module):
    """
    DINOv2-base backbone with a lightweight linear classification head.
    """
    BACKBONE_ID = "facebook/dinov2-base"

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()

        # Load pretrained ViT-B/14 weights from HuggingFace Hub
        self.backbone = Dinov2Model.from_pretrained(self.BACKBONE_ID)
        hidden_size   = self.backbone.config.hidden_size  # 768 for ViT-B

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
        )

    # Backbone freeze / unfreeze helpers

    def freeze_backbone(self):
        """Disable gradient computation for all backbone parameters (Phase 1)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Re-enable gradient computation for all backbone parameters (Phase 2)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    # Forward pass
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs   = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_token)

# Training pass — one epoch
def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    phase: str,
) -> float:
    model.train()
    total_loss  = 0.0
    n_correct   = 0
    n_samples   = 0

    # Inner progress bar
    desc = f"  E{epoch:>2}/{total_epochs} [{phase}]"
    with tqdm(loader, desc=desc, leave=False, unit="batch",
              bar_format="{l_bar}{bar:25}{r_bar}") as pbar:

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            # Forward
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)

            # Backward
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Running stats
            batch_n     = imgs.size(0)
            total_loss += loss.item() * batch_n
            n_correct  += (logits.argmax(1) == labels).sum().item()
            n_samples  += batch_n

            pbar.set_postfix(
                loss=f"{total_loss / n_samples:.4f}",
                acc=f"{n_correct / n_samples:.3f}",
            )

    return total_loss / len(loader.dataset)

# Evaluation pass — one epoch
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    epoch: int,
    total_epochs: int,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    desc = f"  Eval  E{epoch:>2}/{total_epochs}          "
    with tqdm(loader, desc=desc, leave=False, unit="batch",
              bar_format="{l_bar}{bar:25}{r_bar}") as pbar:

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            output = model(imgs)
            logits = output[0] if isinstance(output, tuple) else output
            loss   = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            # Show running accuracy on the eval bar
            running_acc = sum(t == p for t, p in zip(all_labels, all_preds)) / len(all_labels)
            pbar.set_postfix(acc=f"{running_acc:.3f}")

    # Macro-F1 (one-vs-rest per class)
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for true, pred in zip(all_labels, all_preds):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    f1_scores = []
    for c in range(num_classes):
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1_scores.append((2 * tp[c] / denom) if denom > 0 else 0.0)

    macro_f1 = sum(f1_scores) / num_classes
    accuracy  = sum(t == p for t, p in zip(all_labels, all_preds)) / len(all_labels)

    return {
        "loss":      total_loss / len(loader.dataset),
        "accuracy":  accuracy,
        "macro_f1":  macro_f1,
        "preds":     all_preds,
        "labels":    all_labels,
    }

# Main training loop
def train(args):
    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # Adaptation flags
    use_data_adapt = getattr(args, "use_data_adapt", False)
    use_dann       = getattr(args, "use_dann",       False)
    da_strength    = getattr(args, "da_strength",    "medium")
    dann_weight    = getattr(args, "dann_weight",    0.2)

    # Datasets & loaders
    if use_dann:
        # DANN path: build three datasets (source train, test, unlabelled target)
        tqdm.write("  [DANN] Building domain-aware datasets …")
        train_ds, test_ds, target_ds, label2idx, idx2label = build_dann_datasets(
            args.data_root,
            da_strength=da_strength if use_data_adapt else "light",
        )
        train_loader, test_loader, target_loader = get_dann_dataloaders(
            train_ds, test_ds, target_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        # Swap in adapted transforms when --use_data_adapt is set.
        if use_data_adapt:
            tqdm.write(f"  [DataAdapt] Using night-simulation transforms "
                        f"(strength={da_strength}) …")
            from dataset import scan_voc_split, WildlifeDataset
            from pathlib import Path as _Path
            root          = _Path(args.data_root)
            train_records = scan_voc_split(root / "voc_day")
            test_records  = scan_voc_split(root / "voc_night")
            all_labels    = sorted({lbl for _, lbl, _ in train_records})
            label2idx     = {lbl: i for i, lbl in enumerate(all_labels)}
            idx2label     = {i: lbl for lbl, i in label2idx.items()}
            filtered_test = [r for r in test_records if r[1] in label2idx]
            train_ds      = WildlifeDataset(
                train_records, label2idx,
                get_adapted_train_transforms(da_strength),
            )
            test_ds = WildlifeDataset(
                filtered_test, label2idx,
                __import__("data_adaptation").get_night_test_transforms(),
            )
            print(f"\nClasses ({len(all_labels)}): {all_labels}\n")
        else:
            # Baseline dataset build
            train_ds, test_ds, label2idx, idx2label = build_datasets(args.data_root)

        train_loader, test_loader = get_dataloaders(
            train_ds, test_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        target_loader = None

    num_classes = len(label2idx)

    # Model — start with backbone frozen (Phase 1)
    if use_dann:
        # DANN model
        tqdm.write("  [DANN] Building DomainAdversarialModel …")
        model = DomainAdversarialModel(
            num_classes=num_classes, dropout=0.3, lambda_=0.0
        ).to(device)
    else:
        # Baseline model
        model = Dinov2Classifier(num_classes=num_classes, dropout=0.3).to(device)

    model.freeze_backbone()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Phase 1 optimiser: only head parameters have requires_grad=True.
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    # Cosine annealing smoothly reduces LR to near-zero over the full run.
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Pre-compute total steps for lambda annealing
    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * args.epochs
    global_step     = 0   # incremented inside dann_train_one_epoch

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "label_map.json", "w") as f:
        json.dump({
            "label2idx": label2idx,
            "idx2label": {str(k): v for k, v in idx2label.items()},
        }, f, indent=2)

    # CSV training log
    log_path = out_dir / "train_log.csv"
    log_file = open(log_path, "w", newline="")
    logger   = csv.writer(log_file)
    logger.writerow([
        "epoch", "phase",
        "train_loss",
        "dann_cls_loss", "dann_domain_loss",
        "test_loss", "test_acc", "test_macro_f1",
    ])

    best_f1    = -1.0
    best_epoch = -1
    best_path  = out_dir / "best_model.pt"

    # Header
    print(f"\n{'='*60}")
    print(f"  DINOv2-base  |  {num_classes} classes  |  "
          f"{len(train_ds)} train / {len(test_ds)} test")
    print(f"  Epochs: {args.epochs}  |  Warmup: {args.warmup_epochs}  |  "
          f"LR: {args.lr}  |  Batch: {args.batch_size}")
    # Print active adaptation modules
    adapt_flags = []
    if use_data_adapt:
        adapt_flags.append(f"DataAdapt({da_strength})")
    if use_dann:
        adapt_flags.append(f"DANN(w={dann_weight})")
    print(f"  Adaptation:  {', '.join(adapt_flags) if adapt_flags else 'none (baseline)'}")
    print(f"{'='*60}\n")

    # Tracks overall progress across all epochs.
    epoch_bar = tqdm(
        range(1, args.epochs + 1),
        desc="  Epochs",
        unit="ep",
        bar_format="{l_bar}{bar:30}{r_bar}",
    )

    for epoch in epoch_bar:
        # After warmup_epochs the backbone is unfrozen and the optimiser is re-initialised.
        if epoch == args.warmup_epochs + 1:
            tqdm.write("\n── Phase 2: unfreezing backbone ──")
            model.unfreeze_backbone()

            # Rebuild optimiser so backbone params get a 10× smaller LR.
            optimizer = AdamW(
                model.parameters(),
                lr=args.lr * 0.1,
                weight_decay=1e-4,
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - args.warmup_epochs,
            )

        phase = "warmup" if epoch <= args.warmup_epochs else "finetune"
        t0    = time.time()

        # Train
        dann_cls_loss    = ""
        dann_domain_loss = ""

        if use_dann:
            # DANN training step
            dann_cls, dann_dom, train_loss, global_step = dann_train_one_epoch(
                model         = model,
                source_loader = train_loader,
                target_loader = target_loader,
                optimizer     = optimizer,
                cls_criterion = criterion,
                device        = device,
                epoch         = epoch,
                total_epochs  = args.epochs,
                phase         = phase,
                global_step   = global_step,
                total_steps   = total_steps,
                dann_weight   = dann_weight,
            )
            dann_cls_loss    = f"{dann_cls:.6f}"
            dann_domain_loss = f"{dann_dom:.6f}"
        else:
            # Standard single-domain training step
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion,
                device, epoch, args.epochs, phase,
            )

        # Evaluate
        test_metrics = evaluate(model, test_loader, criterion, device,
                                num_classes, epoch, args.epochs)
        scheduler.step()

        elapsed = time.time() - t0

        # Update outer bar postfix
        postfix = dict(
            phase=phase,
            tr_loss=f"{train_loss:.4f}",
            te_loss=f"{test_metrics['loss']:.4f}",
            acc=f"{test_metrics['accuracy']:.3f}",
            F1=f"{test_metrics['macro_f1']:.3f}",
        )
        epoch_bar.set_postfix(**postfix)

        dann_str = ""
        if use_dann and dann_cls_loss:
            lam = compute_lambda(global_step, total_steps)
            dann_str = (f"  cls={dann_cls_loss}  dom={dann_domain_loss}"
                        f"  λ={lam:.3f}")
        tqdm.write(
            f"  Epoch {epoch:>3}/{args.epochs} [{phase:>8}]  "
            f"train_loss={train_loss:.4f}{dann_str}  "
            f"test_loss={test_metrics['loss']:.4f}  "
            f"acc={test_metrics['accuracy']:.4f}  "
            f"macro_F1={test_metrics['macro_f1']:.4f}  "
            f"({elapsed:.0f}s)"
        )

        # CSV log
        logger.writerow([
            epoch, phase,
            f"{train_loss:.6f}",
            dann_cls_loss,
            dann_domain_loss,
            f"{test_metrics['loss']:.6f}",
            f"{test_metrics['accuracy']:.6f}",
            f"{test_metrics['macro_f1']:.6f}",
        ])
        log_file.flush()

        # Checkpoint — save whenever test macro-F1 improves
        if test_metrics["macro_f1"] > best_f1:
            best_f1    = test_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "label2idx":   label2idx,
                    "idx2label":   idx2label,
                    "num_classes": num_classes,
                    "args":        vars(args),
                    "use_dann":     use_dann,
                    "use_data_adapt": use_data_adapt,
                },
                best_path,
            )
            tqdm.write(f" Best model saved (macro-F1 = {best_f1:.4f})")

    log_file.close()

    tqdm.write(
        f"\n Training complete. Best macro-F1 = {best_f1:.4f} at epoch {best_epoch}."
        f"\n Checkpoint : {best_path}"
        f"\n Log        : {log_path}"
    )

# Entry point
def parse_args():
    p = argparse.ArgumentParser(description="Train DINOv2-base on camera-trap images.")
    p.add_argument("--data_root",     default="data",    help="Root dir with voc_day/ and voc_night/")
    p.add_argument("--output_dir",    default="outputs", help="Where to save checkpoints and logs")
    p.add_argument("--epochs",        type=int,   default=20,   help="Total training epochs")
    p.add_argument("--warmup_epochs", type=int,   default=5,    help="Epochs with backbone frozen")
    p.add_argument("--batch_size",    type=int,   default=16,   help="Batch size")
    p.add_argument("--lr",            type=float, default=1e-4, help="Head learning rate")
    p.add_argument("--num_workers",   type=int,   default=4,    help="DataLoader worker processes")
    # Data-level adaptation arguments
    p.add_argument("--use_data_adapt", action="store_true",
                    help="Apply night-simulation augmentations to training images")
    p.add_argument("--da_strength",   default="medium",
                    choices=["light", "medium", "strong"],
                    help="Intensity of night-simulation augmentation")
    # Feature-level DANN arguments
    p.add_argument("--use_dann",      action="store_true",
                    help="Enable domain-adversarial training (DANN)")
    p.add_argument("--dann_weight",   type=float, default=0.2,
                    help="Scale factor on the DANN domain loss")
    return p.parse_args()

if __name__ == "__main__":
    train(parse_args())
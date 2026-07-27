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
from data_adaptation import get_adapted_train_transforms, get_night_test_transforms
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

    def unfreeze_last_blocks(self, n: int = 3):
        """
        Unfreeze only the last `n` transformer encoder blocks and the layer norm that follows them.  All other backbone parameters remain frozen.
        """
        # Keep everything frozen first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last n encoder blocks
        total_blocks = len(self.backbone.encoder.layer)
        for block in self.backbone.encoder.layer[total_blocks - n:]:
            for param in block.parameters():
                param.requires_grad = True

        # Unfreeze the final layernorm (sits after all encoder blocks)
        if hasattr(self.backbone, "layernorm"):
            for param in self.backbone.layernorm.parameters():
                param.requires_grad = True

    # Forward pass
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs   = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_token)

# Training pass — one epoch
def train_one_epoch(
    model, loader, optimizer, criterion, device,
    epoch, total_epochs, phase,
) -> float:
    model.train()
    total_loss = 0.0
    n_correct  = 0
    n_samples  = 0

    desc = f"  Train E{epoch:>2}/{total_epochs} [{phase}]"
    with tqdm(loader, desc=desc, leave=False, unit="batch",
              bar_format="{l_bar}{bar:25}{r_bar}") as pbar:
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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
    model, loader, criterion, device,
    num_classes, epoch, total_epochs, split_name="Eval",
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    desc = f"  {split_name:<8} E{epoch:>2}/{total_epochs}"
    with tqdm(loader, desc=desc, leave=False, unit="batch",
              bar_format="{l_bar}{bar:25}{r_bar}") as pbar:
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            output = model(imgs)
            logits = output[0] if isinstance(output, tuple) else output

            loss = criterion(logits, labels)
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
    }

# Main training loop
def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    use_data_adapt  = getattr(args, "use_data_adapt",  False)
    use_dann        = getattr(args, "use_dann",        False)
    da_strength     = getattr(args, "da_strength",     "medium")
    dann_weight     = getattr(args, "dann_weight",     1.0)
    finetune_blocks = getattr(args, "finetune_blocks", 3)
    run_ts          = getattr(args, "run_ts",          "run")

    # Datasets & loaders
    if use_dann:
        tqdm.write("  [DANN] Building domain-aware datasets …")
        train_ds, val_day_ds, test_ds, val_night_ds, label2idx, idx2label = \
            build_dann_datasets(
                args.data_root,
                da_strength=da_strength if use_data_adapt else "light",
            )
        train_loader, val_day_loader, test_loader, val_night_loader = \
            get_dann_dataloaders(
                train_ds, val_day_ds, test_ds, val_night_ds,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
        # Build unlabelled target loader separately (all night images)
        from dann import DomainDataset
        from pathlib import Path as _Path
        night_img_dir = _Path(args.data_root) / "voc_night" / "JPEGImages"
        target_ds     = DomainDataset.from_dir(night_img_dir, get_night_test_transforms())
        from torch.utils.data import DataLoader as _DL
        target_loader = _DL(
            target_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
    else:
        # Swap in adapted transforms when --use_data_adapt is set.
        if use_data_adapt:
            tqdm.write(f"  [DataAdapt] strength={da_strength} …")
            from dataset import scan_voc_split, WildlifeDataset, stratified_split, \
                get_eval_transforms, SPLIT_SEED
            from pathlib import Path as _Path
            root          = _Path(args.data_root)
            day_records   = scan_voc_split(root / "voc_day")
            night_records = scan_voc_split(root / "voc_night")
            all_labels    = sorted({lbl for _, lbl, _ in day_records})
            label2idx     = {lbl: i for i, lbl in enumerate(all_labels)}
            idx2label     = {i: lbl for lbl, i in label2idx.items()}
            print(f"\nClasses ({len(all_labels)}): {all_labels}\n")

            train_records, val_day_records = stratified_split(
                day_records, val_fraction=0.10, seed=SPLIT_SEED
            )
            night_known = [r for r in night_records if r[1] in label2idx]
            test_records, val_night_records = stratified_split(
                night_known, val_fraction=0.20, seed=SPLIT_SEED
            )
            print(f"  voc_day   : {len(train_records)} train / {len(val_day_records)} val")
            print(f"  voc_night : {len(test_records)} test  / {len(val_night_records)} val\n")

            train_ds     = WildlifeDataset(train_records,     label2idx,
                                           get_adapted_train_transforms(da_strength))
            val_day_ds   = WildlifeDataset(val_day_records,   label2idx, get_eval_transforms())
            test_ds      = WildlifeDataset(test_records,      label2idx, get_eval_transforms())
            val_night_ds = WildlifeDataset(val_night_records, label2idx, get_night_test_transforms())
        else:
            train_ds, val_day_ds, test_ds, val_night_ds, label2idx, idx2label = \
                build_datasets(args.data_root)

        train_loader, val_day_loader, test_loader, val_night_loader = get_dataloaders(
            train_ds, val_day_ds, test_ds, val_night_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        target_loader = None

    num_classes = len(label2idx)

    # Model — start with backbone frozen (Phase 1)
    if use_dann:
        tqdm.write("  [DANN] Building DomainAdversarialModel …")
        model = DomainAdversarialModel(
            num_classes=num_classes, dropout=0.3, lambda_=0.0
        ).to(device)
    else:
        model = Dinov2Classifier(num_classes=num_classes, dropout=0.3).to(device)

    model.freeze_backbone()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Phase 1 optimiser: only head parameters have requires_grad=True.
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Pre-compute total steps for lambda annealing
    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * args.epochs
    global_step     = 0

    # Output paths with datetime stamp
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"best_model_{run_ts}.pt"
    log_path  = out_dir / f"train_log_{run_ts}.csv"

    with open(out_dir / f"label_map_{run_ts}.json", "w") as f:
        json.dump({
            "label2idx": label2idx,
            "idx2label": {str(k): v for k, v in idx2label.items()},
        }, f, indent=2)

    log_file = open(log_path, "w", newline="")
    logger   = csv.writer(log_file)
    logger.writerow([
        "epoch", "phase",
        "train_loss",
        "dann_cls_loss", "dann_domain_loss",
        "val_day_loss",   "val_day_acc",   "val_day_f1",
        "val_night_loss", "val_night_acc", "val_night_f1",
    ])

    best_f1    = -1.0
    best_epoch = -1

    # Header
    print(f"\n{'='*60}")
    print(f"  DINOv2-base  |  {num_classes} classes")
    print(f"  Train: {len(train_ds)}  ValDay: {len(val_day_ds)}  "
          f"Test: {len(test_ds)}  ValNight: {len(val_night_ds)}")
    print(f"  Epochs: {args.epochs}  Warmup: {args.warmup_epochs}  "
          f"FineTuneBlocks: {finetune_blocks}  LR: {args.lr}  Batch: {args.batch_size}")
    adapt_flags = []
    if use_data_adapt:
        adapt_flags.append(f"DataAdapt({da_strength})")
    if use_dann:
        adapt_flags.append(f"DANN(w={dann_weight})")
    print(f"  Adaptation: {', '.join(adapt_flags) if adapt_flags else 'none (baseline)'}")
    print(f"{'='*60}\n")

    # Tracks overall progress across all epochs.
    epoch_bar = tqdm(
        range(1, args.epochs + 1),
        desc="  Epochs", unit="ep",
        bar_format="{l_bar}{bar:30}{r_bar}",
    )

    for epoch in epoch_bar:

        # Phase transition: partial unfreeze
        if epoch == args.warmup_epochs + 1:
            tqdm.write(f"\n── Phase 2: unfreezing last {finetune_blocks} blocks ──")
            model.unfreeze_last_blocks(n=finetune_blocks)

            if use_dann:
                global_step = 0
                total_steps = steps_per_epoch * (args.epochs - args.warmup_epochs)
                tqdm.write("  [DANN] λ reset to 0 for Phase 2.")

            # Separate param groups: unfrozen backbone at lr×0.02, head at lr
            backbone_params = [p for p in model.backbone.parameters()
                               if p.requires_grad]
            head_params     = list(model.head.parameters())
            if use_dann:
                head_params += list(model.domain_classifier.parameters())

            optimizer = AdamW(
                [
                    {"params": backbone_params, "lr": args.lr * 0.02},
                    {"params": head_params,     "lr": args.lr},
                ],
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
            dann_cls, dann_dom, train_loss, global_step = dann_train_one_epoch(
                model=model, source_loader=train_loader, target_loader=target_loader,
                optimizer=optimizer, cls_criterion=criterion, device=device,
                epoch=epoch, total_epochs=args.epochs, phase=phase,
                global_step=global_step, total_steps=total_steps,
                dann_weight=dann_weight,
            )
            dann_cls_loss    = f"{dann_cls:.6f}"
            dann_domain_loss = f"{dann_dom:.6f}"
        else:
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion,
                device, epoch, args.epochs, phase,
            )

        # Evaluate on both validation sets
        val_day_m   = evaluate(model, val_day_loader,   criterion, device,
                               num_classes, epoch, args.epochs, "ValDay")
        val_night_m = evaluate(model, val_night_loader, criterion, device,
                               num_classes, epoch, args.epochs, "ValNight")
        scheduler.step()

        elapsed = time.time() - t0

        # Logging
        epoch_bar.set_postfix(
            phase=phase,
            tr=f"{train_loss:.4f}",
            vd_F1=f"{val_day_m['macro_f1']:.3f}",
            vn_F1=f"{val_night_m['macro_f1']:.3f}",
        )

        dann_str = ""
        if use_dann and dann_cls_loss:
            lam = compute_lambda(global_step, total_steps)
            dann_str = f"  cls={dann_cls_loss}  dom={dann_domain_loss}  λ={lam:.3f}"

        tqdm.write(
            f"  Epoch {epoch:>3}/{args.epochs} [{phase:>8}]  "
            f"train_loss={train_loss:.4f}{dann_str}  "
            f"val_day[loss={val_day_m['loss']:.4f} acc={val_day_m['accuracy']:.4f} "
            f"F1={val_day_m['macro_f1']:.4f}]  "
            f"val_night[loss={val_night_m['loss']:.4f} acc={val_night_m['accuracy']:.4f} "
            f"F1={val_night_m['macro_f1']:.4f}]  "
            f"({elapsed:.0f}s)"
        )

        logger.writerow([
            epoch, phase,
            f"{train_loss:.6f}",
            dann_cls_loss, dann_domain_loss,
            f"{val_day_m['loss']:.6f}",   f"{val_day_m['accuracy']:.6f}",   f"{val_day_m['macro_f1']:.6f}",
            f"{val_night_m['loss']:.6f}", f"{val_night_m['accuracy']:.6f}", f"{val_night_m['macro_f1']:.6f}",
        ])
        log_file.flush()

        # Checkpoint
        if val_night_m["macro_f1"] > best_f1:
            best_f1    = val_night_m["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch":          epoch,
                    "model_state":    model.state_dict(),
                    "label2idx":      label2idx,
                    "idx2label":      idx2label,
                    "num_classes":    num_classes,
                    "args":           vars(args),
                    "use_dann":       use_dann,
                    "use_data_adapt": use_data_adapt,
                },
                best_path,
            )
            tqdm.write(f" Best model saved (val_night macro-F1 = {best_f1:.4f})")

    log_file.close()
    tqdm.write(
        f"\nTraining complete.  Best val_night macro-F1 = {best_f1:.4f} at epoch {best_epoch}."
        f"\n  Checkpoint : {best_path}"
        f"\n  Log        : {log_path}"
    )

# Entry point
def parse_args():
    p = argparse.ArgumentParser(description="Train DINOv2-base on camera-trap images.")
    p.add_argument("--data_root",       default="./data")
    p.add_argument("--output_dir",      default="./outputs")
    p.add_argument("--run_ts",          default="run")
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--warmup_epochs",   type=int,   default=15)
    p.add_argument("--finetune_blocks", type=int,   default=2,
                   help="Number of last transformer blocks to unfreeze in Phase 2")
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--use_data_adapt",  action="store_true")
    p.add_argument("--da_strength",     default="medium",
                   choices=["light", "medium", "strong"])
    p.add_argument("--use_dann",        action="store_true")
    p.add_argument("--dann_weight",     type=float, default=1.0)
    return p.parse_args()

if __name__ == "__main__":
    train(parse_args())
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import Dinov2Model

from dataset import build_datasets, get_dataloaders



# Model
class Dinov2Classifier(nn.Module):
    """
    DINOv2-base backbone with a linear classification head and an optional
    projection head for supervised contrastive learning.

    Architecture
    ------------
    Backbone       : DINOv2-base (ViT-B/14, 12 transformer blocks, hidden dim 768).
    Head           : Dropout(p) -> Linear(768, num_classes).
    Projection head: Linear(768, 768) -> ReLU -> Linear(768, 128), L2-normalised.
                     Present only when use_supcon is True.  Used during Phase 1
                     SupCon training and discarded in Phase 2.

    Backbone state per phase
    ------------------------
    Without SupCon:
      Phase 1 : fully frozen.
      Phase 2 : frozen except last finetune_blocks blocks and final layer norm.

    With SupCon:
      Phase 1a : fully frozen (projection head stabilisation).
      Phase 1b : fully unfrozen at lr * 0.01 (contrastive backbone shaping).
      Phase 2  : re-frozen, then last finetune_blocks blocks unfrozen at lr * 0.02.
    """

    BACKBONE_ID = "facebook/dinov2-base"

    def __init__(self, num_classes: int, dropout: float = 0.3,
                 use_supcon: bool = False):
        super().__init__()
        self.use_supcon = use_supcon

        self.backbone = Dinov2Model.from_pretrained(self.BACKBONE_ID)
        hidden_size   = self.backbone.config.hidden_size  # 768

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
        )

        # Projection head used only during SupCon warmup phase.
        if use_supcon:
            self.proj_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, 128),
            )

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters (used in SupCon Phase 1b)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_blocks(self, n: int = 3):
        """
        Unfreeze the last n transformer encoder blocks and the final layer norm.

        All other backbone parameters remain frozen.  For DINOv2-base with 12
        encoder blocks total, n=3 unfreezes blocks 9, 10, and 11.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        total_blocks = len(self.backbone.encoder.layer)
        for block in self.backbone.encoder.layer[total_blocks - n:]:
            for param in block.parameters():
                param.requires_grad = True

        if hasattr(self.backbone, "layernorm"):
            for param in self.backbone.layernorm.parameters():
                param.requires_grad = True

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_projection: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pixel_values      : (B, 3, 224, 224) normalised image batch.
        return_projection : if True, return the L2-normalised projection
                            embedding instead of class logits.  Only valid
                            when use_supcon is True.

        Returns
        -------
        logits     : (B, num_classes) when return_projection is False.
        projection : (B, 128) L2-normalised when return_projection is True.
        """
        outputs   = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]  # (B, 768)

        if return_projection and self.use_supcon:
            proj = self.proj_head(cls_token)
            return F.normalize(proj, dim=1)

        return self.head(cls_token)

# Supervised Contrastive Loss
class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    For each anchor sample, all samples sharing the same class label are
    treated as positives and all others as negatives.  The loss encourages
    the model to produce embeddings where same-class samples cluster together
    on the unit hypersphere.

    Parameters
    ----------
    temperature : float
        Temperature scaling factor applied to the dot-product similarities.
        Lower values sharpen the distribution; 0.07 is the value used in the
        original paper.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features : (B, D) L2-normalised embeddings.
        labels   : (B,)   integer class labels.

        Returns
        -------
        loss : scalar tensor.
        """
        device = features.device
        B      = features.size(0)

        # Pairwise cosine similarity matrix scaled by temperature.
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # Mask: 1 where i and j share the same class, 0 otherwise.
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        # Exclude the diagonal (self-similarity).
        eye      = torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = label_eq & ~eye

        # Numerical stability: subtract row-wise maximum.
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim        = sim - sim_max.detach()

        exp_sim = torch.exp(sim)

        # Denominator: sum over all non-self pairs.
        denom = (exp_sim * (~eye).float()).sum(dim=1)  # (B,)

        # Numerator: sum of exp similarities over positives.
        # For rows with no positives (single-sample classes in the batch),
        # the loss contribution is zero.
        n_pos = pos_mask.float().sum(dim=1)  # (B,)
        log_prob = sim - torch.log(denom + 1e-8)

        loss_per_anchor = -(pos_mask.float() * log_prob).sum(dim=1) / (n_pos + 1e-8)

        # Average only over anchors that have at least one positive.
        valid = n_pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return loss_per_anchor[valid].mean()


# Training pass - one epoch
def train_one_epoch(
    model, loader, optimizer, criterion, device,
    epoch, total_epochs, sub_phase, use_supcon_active,
) -> float:
    """
    Run one full pass over the training DataLoader.

    Parameters
    ----------
    sub_phase        : string label describing the current sub-phase, used in
                       the tqdm progress bar (e.g. "1a", "1b", "2").
    use_supcon_active: whether SupCon loss and the projection head are active
                       for this epoch.  When False, the classification head and
                       cross-entropy are used.

    Returns
    -------
    mean_loss : float  average loss per sample for this epoch.
    """
    model.train()
    total_loss = 0.0
    n_correct  = 0
    n_samples  = 0

    desc = f"  Train E{epoch:>2}/{total_epochs} [phase {sub_phase}]"
    with tqdm(loader, desc=desc, leave=False, unit="batch",
              bar_format="{l_bar}{bar:25}{r_bar}") as pbar:
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            if use_supcon_active:
                embeddings = model(imgs, return_projection=True)
                loss       = criterion(embeddings, labels)
            else:
                logits = model(imgs)
                loss   = criterion(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_n     = imgs.size(0)
            total_loss += loss.item() * batch_n
            n_samples  += batch_n

            if not use_supcon_active:
                n_correct += (logits.argmax(1) == labels).sum().item()
                pbar.set_postfix(
                    loss=f"{total_loss / n_samples:.4f}",
                    acc=f"{n_correct / n_samples:.3f}",
                )
            else:
                pbar.set_postfix(loss=f"{total_loss / n_samples:.4f}")

    return total_loss / len(loader.dataset)

# Evaluation pass - one epoch
@torch.no_grad()
def evaluate(
    model, loader, criterion, device,
    num_classes, epoch, total_epochs, split_name="Eval",
) -> dict:
    """
    Run inference on a DataLoader and return loss, accuracy, and macro-F1.

    Always uses the classification head regardless of SupCon setting.
    """
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    desc = f"  {split_name:<8} E{epoch:>2}/{total_epochs}"
    with tqdm(loader, desc=desc, leave=False, unit="batch",
              bar_format="{l_bar}{bar:25}{r_bar}") as pbar:
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            # Always call the classification head for evaluation.
            logits = model(imgs, return_projection=False)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            running_acc = sum(
                t == p for t, p in zip(all_labels, all_preds)
            ) / len(all_labels)
            pbar.set_postfix(acc=f"{running_acc:.3f}")

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
    accuracy  = sum(
        t == p for t, p in zip(all_labels, all_preds)
    ) / len(all_labels)

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
    use_supcon      = getattr(args, "use_supcon",       False)
    finetune_blocks = getattr(args, "finetune_blocks",  3)
    run_ts          = getattr(args, "run_ts",           "run")

    # Datasets and loaders
    train_ds, test_ds, val_night_ds, label2idx, idx2label = build_datasets(
        args.data_root, use_data_adapt=use_data_adapt
    )
    train_loader, test_loader, val_night_loader = get_dataloaders(
        train_ds, test_ds, val_night_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_classes = len(label2idx)

    # SupCon epoch boundaries.
    # Phase 1a: epochs 1 .. supcon_freeze_end  (backbone frozen, proj head, SupCon)
    # Phase 1b: epochs supcon_freeze_end+1 .. warmup_epochs  (backbone unfrozen, SupCon)
    # Phase 2 : epochs warmup_epochs+1 .. total_epochs  (last N blocks, CE)
    #
    # When SupCon is off, supcon_freeze_end is unused; all warmup_epochs are
    # standard Phase 1 (backbone frozen, classification head, CE).
    SUPCON_HEAD_STABILISE = 5
    if use_supcon:
        supcon_freeze_end = min(SUPCON_HEAD_STABILISE, args.warmup_epochs)
    else:
        supcon_freeze_end = args.warmup_epochs  # unused but defined for clarity

    # Model
    model = Dinov2Classifier(
        num_classes=num_classes, dropout=0.3, use_supcon=use_supcon
    ).to(device)
    model.freeze_backbone()

    # Loss functions.
    ce_criterion     = nn.CrossEntropyLoss(label_smoothing=0.1)
    supcon_criterion = SupConLoss(temperature=0.07)

    # Initial optimiser for epoch 1.
    # Without SupCon: backbone frozen, classification head only.
    # With SupCon:    backbone frozen, projection head only.
    #                 Classification head is excluded entirely until Phase 2.
    if use_supcon:
        initial_params = list(model.proj_head.parameters())
    else:
        initial_params = list(model.head.parameters())

    optimizer = AdamW(initial_params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Output paths
    out_dir   = Path(args.output_dir)
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
        "epoch", "sub_phase", "train_loss",
        "val_night_loss", "val_night_acc", "val_night_f1",
    ])

    best_f1    = -1.0
    best_epoch = -1

    # Training header
    print(f"\n{'='*60}")
    print(f"  Model    : DINOv2-base  ({num_classes} classes)")
    print(f"  Train    : {len(train_ds)} samples, "
          f"Val night : {len(val_night_ds)} samples, "
          f"Test : {len(test_ds)} samples")
    if use_supcon:
        print(f"  Epochs   : {args.epochs} total  ("
              f"phase 1a: {supcon_freeze_end}, "
              f"phase 1b: {max(0, args.warmup_epochs - supcon_freeze_end)}, "
              f"phase 2: {args.epochs - args.warmup_epochs})")
    else:
        print(f"  Epochs   : {args.epochs} total  ("
              f"phase 1: {args.warmup_epochs}, "
              f"phase 2: {args.epochs - args.warmup_epochs})")
    print(f"  LR       : {args.lr}  "
          f"(backbone phase 1b: {args.lr * 0.01}, phase 2: {args.lr * 0.02})")
    print(f"  Batch    : {args.batch_size}")
    adapt_parts = []
    if use_data_adapt:
        adapt_parts.append("data adaptation (medium)")
    if use_supcon:
        adapt_parts.append("supervised contrastive loss")
    print(f"  Options  : {', '.join(adapt_parts) if adapt_parts else 'none (baseline)'}")
    print(f"{'='*60}\n")

    epoch_bar = tqdm(
        range(1, args.epochs + 1),
        desc="  Epochs", unit="ep",
        bar_format="{l_bar}{bar:30}{r_bar}",
    )

    for epoch in epoch_bar:

        # Determine sub-phase and apply any transitions

        if not use_supcon:
            # Standard two-phase schedule.
            if epoch == args.warmup_epochs + 1:
                # Transition: Phase 1 -> Phase 2.
                tqdm.write(
                    f"\nPhase 2: unfreezing last {finetune_blocks} encoder blocks."
                )
                model.unfreeze_last_blocks(n=finetune_blocks)
                backbone_params = [p for p in model.backbone.parameters()
                                   if p.requires_grad]
                head_params     = list(model.head.parameters())
                optimizer = AdamW(
                    [
                        {"params": backbone_params, "lr": args.lr * 0.02},
                        {"params": head_params,     "lr": args.lr},
                    ],
                    weight_decay=1e-4,
                )
                scheduler = CosineAnnealingLR(
                    optimizer, T_max=args.epochs - args.warmup_epochs
                )
            sub_phase       = "1" if epoch <= args.warmup_epochs else "2"
            use_supcon_now  = False
            active_crit     = ce_criterion

        else:
            # SupCon three-sub-stage schedule.
            if epoch == supcon_freeze_end + 1 and supcon_freeze_end < args.warmup_epochs:
                # Transition: Phase 1a -> Phase 1b.
                # Unfreeze full backbone at a small LR; projection head at full LR.
                tqdm.write(
                    f"\nPhase 1b: backbone fully unfrozen for contrastive shaping "
                    f"(backbone LR = {args.lr * 0.01})."
                )
                model.unfreeze_backbone()
                backbone_params = list(model.backbone.parameters())
                proj_params     = list(model.proj_head.parameters())
                # Classification head still excluded — not trained until Phase 2.
                optimizer = AdamW(
                    [
                        {"params": backbone_params, "lr": args.lr * 0.01},
                        {"params": proj_params,     "lr": args.lr},
                    ],
                    weight_decay=1e-4,
                )
                # Re-initialise scheduler over the remaining warmup epochs.
                remaining_warmup = args.warmup_epochs - supcon_freeze_end
                scheduler = CosineAnnealingLR(
                    optimizer, T_max=max(remaining_warmup, 1)
                )

            elif epoch == args.warmup_epochs + 1:
                # Transition: Phase 1b -> Phase 2.
                # Re-freeze backbone, then unfreeze only the last N blocks.
                tqdm.write(
                    f"\nPhase 2: re-freezing backbone, unfreezing last "
                    f"{finetune_blocks} encoder blocks. Switching to cross-entropy."
                )
                model.unfreeze_last_blocks(n=finetune_blocks)
                backbone_params = [p for p in model.backbone.parameters()
                                   if p.requires_grad]
                # Classification head enters training for the first time.
                # Projection head excluded — discarded after Phase 1.
                head_params = list(model.head.parameters())
                optimizer = AdamW(
                    [
                        {"params": backbone_params, "lr": args.lr * 0.02},
                        {"params": head_params,     "lr": args.lr},
                    ],
                    weight_decay=1e-4,
                )
                scheduler = CosineAnnealingLR(
                    optimizer, T_max=args.epochs - args.warmup_epochs
                )

            if epoch <= supcon_freeze_end:
                sub_phase      = "1a"
                use_supcon_now = True
                active_crit    = supcon_criterion
            elif epoch <= args.warmup_epochs:
                sub_phase      = "1b"
                use_supcon_now = True
                active_crit    = supcon_criterion
            else:
                sub_phase      = "2"
                use_supcon_now = False
                active_crit    = ce_criterion

        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, active_crit,
            device, epoch, args.epochs, sub_phase, use_supcon_now,
        )

        # Evaluation always uses cross-entropy for consistent loss reporting.
        val_night_m = evaluate(
            model, val_night_loader, ce_criterion,
            device, num_classes, epoch, args.epochs, "ValNight",
        )
        scheduler.step()

        elapsed = time.time() - t0

        epoch_bar.set_postfix(
            phase=sub_phase,
            tr=f"{train_loss:.4f}",
            vn_F1=f"{val_night_m['macro_f1']:.3f}",
        )

        tqdm.write(
            f"  Epoch {epoch:>3}/{args.epochs} [phase {sub_phase}]  "
            f"train_loss={train_loss:.4f}  "
            f"val_night(loss={val_night_m['loss']:.4f}  "
            f"acc={val_night_m['accuracy']:.4f}  "
            f"F1={val_night_m['macro_f1']:.4f})  "
            f"({elapsed:.0f}s)"
        )

        logger.writerow([
            epoch, sub_phase,
            f"{train_loss:.6f}",
            f"{val_night_m['loss']:.6f}",
            f"{val_night_m['accuracy']:.6f}",
            f"{val_night_m['macro_f1']:.6f}",
        ])
        log_file.flush()

        # Checkpoint only in phase 2 or when SupCon is off (phase 1 or 2).
        # During SupCon phases 1a/1b the classification head is not being
        # trained, so val_night_f1 is not yet meaningful for checkpointing.
        should_checkpoint = (sub_phase == "2") or (not use_supcon)
        if should_checkpoint and val_night_m["macro_f1"] > best_f1:
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
                    "use_supcon":     use_supcon,
                    "use_data_adapt": use_data_adapt,
                },
                best_path,
            )
            tqdm.write(
                f"  Best model saved at epoch {epoch} "
                f"(val_night macro-F1 = {best_f1:.4f})."
            )

    log_file.close()
    tqdm.write(
        f"\nTraining complete. Best val_night macro-F1 = {best_f1:.4f} "
        f"at epoch {best_epoch}."
        f"\n  Checkpoint : {best_path}"
        f"\n  Log        : {log_path}"
    )

# Entry point
def parse_args():
    p = argparse.ArgumentParser(
        description="Train DINOv2-base on camera-trap images."
    )
    p.add_argument("--data_root",       default="./data")
    p.add_argument("--output_dir",      default="./outputs")
    p.add_argument("--run_ts",          default="run")
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--warmup_epochs",   type=int,   default=15)
    p.add_argument("--finetune_blocks", type=int,   default=3,
                   help="Number of trailing encoder blocks to unfreeze in Phase 2.")
    p.add_argument("--batch_size",      type=int,   default=64)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--use_data_adapt",  action="store_true",
                   help="Append night-simulated copies of daytime images to the "
                        "training set.")
    p.add_argument("--use_supcon",      action="store_true",
                   help="Use supervised contrastive loss during the warmup phase "
                        "instead of cross-entropy.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
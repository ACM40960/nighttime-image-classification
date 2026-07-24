import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Dinov2Model
from PIL import Image

from data_adaptation import get_adapted_train_transforms, get_night_test_transforms

# Gradient Reversal Layer
class GradientReversalFunction(torch.autograd.Function):
    """
    Custom autograd function that implements the gradient reversal layer.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_: float):
        """Update the reversal scale (called each batch from the train loop)."""
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)

    def extra_repr(self) -> str:
        return f"lambda_={self.lambda_:.4f}"


# Lambda schedule
def compute_lambda(current_step: int, total_steps: int,
                   gamma: float = 10.0) -> float:
    """
    Compute the gradient reversal scale λ at a given training step.
    """
    import math
    p = current_step / max(total_steps, 1)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0

# Domain-adversarial model
class DomainAdversarialModel(nn.Module):
    """
    DINOv2-base backbone extended with a domain classifier and GRL.
    """
    BACKBONE_ID = "facebook/dinov2-base"

    def __init__(self, num_classes: int, dropout: float = 0.3,
                 lambda_: float = 0.0):
        super().__init__()

        # Shared backbone
        self.backbone  = Dinov2Model.from_pretrained(self.BACKBONE_ID)
        hidden_size    = self.backbone.config.hidden_size  # 768

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
        )

        # Gradient reversal layer
        self.grl = GradientReversalLayer(lambda_=lambda_)

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )

    # Backbone freeze / unfreeze
    def freeze_backbone(self):
        """Freeze backbone weights (Phase 1)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone weights (Phase 2)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    # GRL scale setter
    def set_lambda(self, lambda_: float):
        """Update the gradient reversal scale.  Called each batch."""
        self.grl.set_lambda(lambda_)

    # Forward pass
    def forward(
        self,
        pixel_values: torch.Tensor,
        return_domain_logit: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Shared feature extraction
        outputs   = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]

        # Classification branch
        cls_logits = self.head(cls_token)

        if not return_domain_logit:
            return cls_logits, None

        # Domain adversarial branch
        reversed_features = self.grl(cls_token)
        domain_logits     = self.domain_classifier(reversed_features)

        return cls_logits, domain_logits

# Domain-aware dataset (unlabelled target images)
class DomainDataset(Dataset):
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, image_paths: List[Path], transform):
        self.image_paths = image_paths
        self.transform   = transform

    @classmethod
    def from_dir(cls, img_dir: Path, transform) -> "DomainDataset":
        """Scan img_dir for all images and return a DomainDataset."""
        paths = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in cls.IMAGE_EXTENSIONS
        )
        if len(paths) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}")
        return cls(paths, transform)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

# Build DANN-aware datasets and loaders
def build_dann_datasets(
    data_root: str,
    da_strength: str = "medium",
) -> Tuple:
    """
    Build the four datasets needed for DANN training
    """
    from dataset import scan_voc_split, WildlifeDataset

    root = Path(data_root)

    # Source domain: daytime labelled images
    train_records = scan_voc_split(root / "voc_day")
    test_records  = scan_voc_split(root / "voc_night")

    all_labels = sorted({label for _, label, _ in train_records})
    label2idx  = {lbl: i for i, lbl in enumerate(all_labels)}
    idx2label  = {i: lbl for lbl, i in label2idx.items()}

    print(f"\nClasses ({len(all_labels)}): {all_labels}\n")

    filtered_test = [r for r in test_records if r[1] in label2idx]

    # Training images use the adapted (night-simulated) transform
    train_ds = WildlifeDataset(
        train_records, label2idx, get_adapted_train_transforms(da_strength)
    )
    # Test images use the plain normalisation transform
    test_ds = WildlifeDataset(
        filtered_test, label2idx, get_night_test_transforms()
    )

    # Target domain: nighttime images
    night_img_dir = root / "voc_night" / "JPEGImages"
    target_ds     = DomainDataset.from_dir(night_img_dir, get_night_test_transforms())

    print(f"  Source (day)  train : {len(train_ds)}")
    print(f"  Target (night) test : {len(test_ds)}")
    print(f"  Target unlabelled   : {len(target_ds)}")

    return train_ds, test_ds, target_ds, label2idx, idx2label


def get_dann_dataloaders(
    train_ds, test_ds, target_ds,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    # Drop last to keep batch sizes consistent when source and target have different lengths.
    target_loader = DataLoader(
        target_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    return train_loader, test_loader, target_loader
 
# DANN training step — one epoch
def dann_train_one_epoch(
    model: DomainAdversarialModel,
    source_loader: DataLoader,
    target_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cls_criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    phase: str,
    global_step: int,
    total_steps: int,
    dann_weight: float = 1.0,
) -> Tuple[float, float, float, int]:
    """
    Run one epoch of DANN training.
    """
    model.train()

    # Cycle the target loader if it is shorter than the source loader
    target_iter = iter(target_loader)

    total_cls_loss    = 0.0
    total_domain_loss = 0.0
    total_loss_sum    = 0.0
    n_samples         = 0
    n_correct         = 0

    # Binary cross-entropy for domain classification (day=0, night=1)
    domain_criterion = nn.BCEWithLogitsLoss()

    desc = f"  DANN  E{epoch:>2}/{total_epochs} [{phase}]"
    with tqdm(source_loader, desc=desc, leave=False, unit="batch",
              bar_format="{l_bar}{bar:25}{r_bar}") as pbar:

        for src_imgs, src_labels in pbar:
            src_imgs   = src_imgs.to(device)
            src_labels = src_labels.to(device)
            batch_n    = src_imgs.size(0)

            # Get target batch (cycle if exhausted)
            try:
                tgt_imgs = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_imgs    = next(target_iter)

            # Ensure target batch has the same size as source batch by slicing
            tgt_imgs = tgt_imgs[:batch_n].to(device)

            # Update lambda (gradient reversal scale)
            if phase == "warmup":
                lam = 0.0
            else:
                lam = compute_lambda(global_step, total_steps)
            model.set_lambda(lam)

            optimizer.zero_grad()

            # Forward: source images
            src_cls_logits, src_domain_logits = model(
                src_imgs, return_domain_logit=True
            )

            # Forward: target images
            _, tgt_domain_logits = model(
                tgt_imgs, return_domain_logit=True
            )

            # Classification loss (source domain only)
            cls_loss = cls_criterion(src_cls_logits, src_labels)

            # Domain loss (source=0, target=1, combined batch)
            # Domain labels: source → 0.0, target → 1.0
            src_domain_labels = torch.zeros(src_domain_logits.size(0), 1,
                                            device=device)
            tgt_domain_labels = torch.ones(tgt_domain_logits.size(0), 1,
                                           device=device)

            domain_logits_all = torch.cat([src_domain_logits,
                                           tgt_domain_logits], dim=0)
            domain_labels_all = torch.cat([src_domain_labels,
                                           tgt_domain_labels], dim=0)
            domain_loss = domain_criterion(domain_logits_all, domain_labels_all)

            # Total loss
            total_loss = cls_loss + dann_weight * lam * domain_loss

            # Backward
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate stats
            total_cls_loss    += cls_loss.item()    * batch_n
            total_domain_loss += domain_loss.item() * batch_n
            total_loss_sum    += total_loss.item()  * batch_n
            n_correct         += (src_cls_logits.argmax(1) == src_labels).sum().item()
            n_samples         += batch_n
            global_step       += 1

            pbar.set_postfix(
                cls=f"{total_cls_loss / n_samples:.4f}",
                dom=f"{total_domain_loss / n_samples:.4f}",
                acc=f"{n_correct / n_samples:.3f}",
                lam=f"{lam:.3f}",
            )

    n = len(source_loader.dataset)
    return (
        total_cls_loss    / n,
        total_domain_loss / n,
        total_loss_sum    / n,
        global_step,
    )
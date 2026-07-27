import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ImageNet statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# DINOv2-base crops
IMAGE_SIZE = 224

# Fractional padding added around each bounding box before cropping.
BBOX_PAD_FRAC = 0.05
SPLIT_SEED = 42
BBox = Optional[Tuple[int, int, int, int]]


# Transforms
def get_train_transforms() -> transforms.Compose:
    """Augmentation-heavy pipeline for the daytime training split."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_eval_transforms() -> transforms.Compose:
    """Deterministic pipeline for validation and test splits."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_test_transforms() -> transforms.Compose:
    """Minimal, deterministic pipeline for the nighttime test split."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# XML parsing
def parse_voc_annotation(xml_path: Path) -> Optional[str]:
    """
    Return the first species label found in a Pascal VOC XML annotation.

    Returns None if the file is malformed or contains no <object> tags.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj = root.find("object")
        if obj is None:
            return None, None
        
        # Species labels
        name_el = obj.find("name")
        if name_el is None or name_el.text is None:
            return None, None
        label =  name_el.text.strip()

        # Bounding box
        bndbox = obj.find("bndbox")
        bbox   = None
        if bndbox is not None:
            try:
                xmin = int(float(bndbox.findtext("xmin")))
                ymin = int(float(bndbox.findtext("ymin")))
                xmax = int(float(bndbox.findtext("xmax")))
                ymax = int(float(bndbox.findtext("ymax")))
                # Box must have positive area
                if xmax > xmin and ymax > ymin:
                    bbox = (xmin, ymin, xmax, ymax)
            except (TypeError, ValueError):
                pass # else return to full image
 
        return label, bbox
    except ET.ParseError:
        return None, None

# Dataset scanning
def scan_voc_split(split_dir: Path) -> List[Tuple[Path, str]]:
    """
    Walk a voc_<split> directory and return a list of (image_path, label).

    Only images that have a matching XML annotation are included.
    Annotations without a discoverable image file are skipped with a warning.
    """
    ann_dir = split_dir / "Annotations"
    img_dir = split_dir / "JPEGImages"

    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {ann_dir}")
    if not img_dir.exists():
        raise FileNotFoundError(f"JPEGImages directory not found: {img_dir}")

    records: List[Tuple[Path, str, "BBox"]] = []
    skipped    = 0
    no_bbox    = 0

    for xml_file in sorted(ann_dir.glob("*.xml")):
        label, bbox = parse_voc_annotation(xml_file)
        if label is None:
            skipped += 1
            continue

        stem = xml_file.stem
        img_path = None
        for ext in IMAGE_EXTENSIONS:
            candidate = img_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"[WARN] No image found for annotation: {xml_file.name}")
            skipped += 1
            continue

        if bbox is None:
            no_bbox += 1

        records.append((img_path, label, bbox))

    print(f"[{split_dir.name}] Loaded {len(records)} samples "
          f"({skipped} skipped, {no_bbox} without bounding boxes).")
    return records

def stratified_split(
    records: List[Tuple[Path, str, BBox]],
    val_fraction: float,
    seed: int = SPLIT_SEED,
) -> Tuple[List, List]:
    """
    Split records into (main, val) lists with class-balanced sampling.
    """
    rng = random.Random(seed)

    # Group indices by class label
    class_indices: Dict[str, List[int]] = defaultdict(list)
    for i, (_, label, _) in enumerate(records):
        class_indices[label].append(i)

    main_idx = []
    val_idx  = []

    for label, indices in sorted(class_indices.items()):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))
        val_idx.extend(shuffled[:n_val])
        main_idx.extend(shuffled[n_val:])

    main_records = [records[i] for i in sorted(main_idx)]
    val_records  = [records[i] for i in sorted(val_idx)]
    return main_records, val_records

# PyTorch Dataset
class WildlifeDataset(Dataset):
    """
    Image-classification dataset built from Pascal VOC annotations.

    Parameters:
        records   : list of (image_path, raw_label) tuples
        label2idx : mapping from string label → integer class index
        transform : torchvision transform applied to each PIL image
    """
    def __init__(
        self,
        records: List[Tuple[Path, str, "BBox"]],
        label2idx: Dict[str, int],
        transform: transforms.Compose,
    ):
        self.records   = records
        self.label2idx = label2idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _crop_to_bbox(image: Image.Image, bbox: "BBox") -> Image.Image:
        """
        Crop image to the bounding box with padding, clamped to image bounds.
        """
        if bbox is None:
            return image   # fallback: return full image unchanged
 
        xmin, ymin, xmax, ymax = bbox
        w, h = image.size
 
        # Compute padding as a fraction of the shorter bbox side
        pad = int(BBOX_PAD_FRAC * min(xmax - xmin, ymax - ymin))
 
        # Expand box by pad, then clamp to image dimensions
        x0 = max(0, xmin - pad)
        y0 = max(0, ymin - pad)
        x1 = min(w, xmax + pad)
        y1 = min(h, ymax + pad)
 
        return image.crop((x0, y0, x1, y1))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label, bbox = self.records[idx]
        image = Image.open(img_path).convert("RGB")
        image = self._crop_to_bbox(image, bbox)
        image = self.transform(image)
        return image, self.label2idx[label]

# Build datasets and dataloaders
def build_datasets(
    data_root: str = "data",
) -> Tuple["WildlifeDataset", "WildlifeDataset",
           "WildlifeDataset", "WildlifeDataset",
           Dict[str, int], Dict[int, str]]:
    """
    Scan voc_day and voc_night, apply class-balanced splitting, and return dataset objects.
    """
    root = Path(data_root)
    day_records   = scan_voc_split(root / "voc_day")
    night_records = scan_voc_split(root / "voc_night")

    # Build vocabulary from voc_day only
    all_labels = sorted({label for _, label, _ in day_records})
    label2idx  = {lbl: i for i, lbl in enumerate(all_labels)}
    idx2label  = {i: lbl for lbl, i in label2idx.items()}

    print(f"\nClasses ({len(all_labels)}): {all_labels}\n")

    # Class-balanced stratified split — voc_day: 90% train / 10% val
    train_records, val_day_records = stratified_split(
        day_records, val_fraction=0.10, seed=SPLIT_SEED
    )

    # Class-balanced stratified split — voc_night: 80% test / 20% val
    night_records_known = [r for r in night_records if r[1] in label2idx]
    test_records, val_night_records = stratified_split(
        night_records_known, val_fraction=0.20, seed=SPLIT_SEED
    )

    print(f"  voc_day   : {len(train_records)} train  / {len(val_day_records)} val")
    print(f"  voc_night : {len(test_records)} test   / {len(val_night_records)} val\n")

    train_ds     = WildlifeDataset(train_records,     label2idx, get_train_transforms())
    val_day_ds   = WildlifeDataset(val_day_records,   label2idx, get_eval_transforms())
    test_ds      = WildlifeDataset(test_records,      label2idx, get_eval_transforms())
    val_night_ds = WildlifeDataset(val_night_records, label2idx, get_eval_transforms())

    return train_ds, val_day_ds, test_ds, val_night_ds, label2idx, idx2label


def get_dataloaders(
    train_ds:     "WildlifeDataset",
    val_day_ds:   "WildlifeDataset",
    test_ds:      "WildlifeDataset",
    val_night_ds: "WildlifeDataset",
    batch_size:   int = 16,
    num_workers:  int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Return train, val_day, test, and val_night DataLoaders."""
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_day_loader = DataLoader(
        val_day_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    val_night_loader = DataLoader(
        val_night_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_day_loader, test_loader, val_night_loader

# Quick self-test
if __name__ == "__main__":
    import sys

    data_root = sys.argv[1] if len(sys.argv) > 1 else "data"
    train_ds, val_day_ds, test_ds, val_night_ds, label2idx, idx2label = build_datasets(data_root)

    print(f"Train samples : {len(train_ds)}")
    print(f"Val (day) samples : {len(val_day_ds)}")
    print(f"Test samples  : {len(test_ds)}")
    print(f"Val (night) samples : {len(val_night_ds)}")
    print(f"Num classes   : {len(label2idx)}")

    # One batch check
    train_loader, val_day_loader, test_loader, val_night_loader = get_dataloaders(
        train_ds, val_day_ds, test_ds, val_night_ds, batch_size=4, num_workers=0
    )
    imgs, labels = next(iter(train_loader))
    print(f"Batch shape   : {imgs.shape}")
    print(f"Label indices : {labels.tolist()}")
    print(f"Label names   : {[idx2label[i.item()] for i in labels]}")
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from torchvision import transforms



# Constants
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMAGE_SIZE = 224

# Fractional padding added around each bounding box before cropping.
BBOX_PAD_FRAC = 0.05

# Reproducibility seed for the night split.
SPLIT_SEED = 42

# Type alias for bounding box coordinates.
BBox = Optional[Tuple[int, int, int, int]]

# Default data root.
DATA_ROOT = "./data"



# Transforms
def get_train_transforms() -> transforms.Compose:
    """Standard augmentation pipeline for daytime training images."""
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
    """Deterministic pipeline for validation and test images."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])



# XML parsing
def parse_voc_annotation(xml_path: Path) -> Tuple[Optional[str], BBox]:
    """
    Parse a Pascal VOC XML annotation file.

    Returns (label, bbox) where bbox is (xmin, ymin, xmax, ymax) in pixel
    coordinates, or None if no valid bounding box is present.
    Returns (None, None) if the file has no object element.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    obj  = root.find("object")
    if obj is None:
        return None, None

    name_el = obj.find("name")
    if name_el is None or name_el.text is None:
        return None, None
    label = name_el.text.strip()

    bndbox = obj.find("bndbox")
    bbox   = None
    if bndbox is not None:
        xmin = int(float(bndbox.findtext("xmin")))
        ymin = int(float(bndbox.findtext("ymin")))
        xmax = int(float(bndbox.findtext("xmax")))
        ymax = int(float(bndbox.findtext("ymax")))
        if xmax > xmin and ymax > ymin:
            bbox = (xmin, ymin, xmax, ymax)

    return label, bbox



# Dataset scanning
def scan_voc_split(split_dir: Path) -> List[Tuple[Path, str, BBox]]:
    """
    Scan a voc split directory and return a list of (image_path, label, bbox)
    triples.  Annotations with no matching image file are skipped.
    """
    ann_dir = split_dir / "Annotations"
    img_dir = split_dir / "JPEGImages"

    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {ann_dir}")
    if not img_dir.exists():
        raise FileNotFoundError(f"JPEGImages directory not found: {img_dir}")

    records: List[Tuple[Path, str, BBox]] = []
    skipped = 0

    for xml_file in sorted(ann_dir.glob("*.xml")):
        label, bbox = parse_voc_annotation(xml_file)
        if label is None:
            skipped += 1
            continue

        stem     = xml_file.stem
        img_path = None
        for ext in IMAGE_EXTENSIONS:
            candidate = img_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"[WARNING] No image found for annotation: {xml_file.name}")
            skipped += 1
            continue

        records.append((img_path, label, bbox))

    n_bbox = sum(1 for _, _, b in records if b is not None)
    print(f"[{split_dir.name}] Loaded {len(records)} samples "
          f"({skipped} skipped, {n_bbox}/{len(records)} with bounding box).")
    return records



# Class-balanced stratified split
def stratified_split(
    records: List[Tuple[Path, str, BBox]],
    val_fraction: float,
    seed: int = SPLIT_SEED,
) -> Tuple[List, List]:
    """
    Split records into (main, val) with class-balanced sampling.

    Each species contributes floor(n * val_fraction) samples to the
    validation set.  The selection within each class is seeded for
    reproducibility.

    Parameters
    ----------
    records      : list of (image_path, label, bbox) triples
    val_fraction : fraction of each class to reserve for validation
    seed         : random seed

    Returns
    -------
    main_records : list
    val_records  : list
    """
    rng = random.Random(seed)

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

    Each sample is cropped to its bounding box before the transform pipeline
    is applied, removing domain-sensitive background.

    Parameters
    ----------
    records   : list of (image_path, label, bbox) triples
    label2idx : mapping from species name to integer class index
    transform : torchvision transform applied after bounding box crop
    """

    def __init__(
        self,
        records: List[Tuple[Path, str, BBox]],
        label2idx: Dict[str, int],
        transform: transforms.Compose,
    ):
        self.records   = records
        self.label2idx = label2idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _crop_to_bbox(image: Image.Image, bbox: BBox) -> Image.Image:
        """
        Crop the image to the bounding box with fractional padding.

        Padding is BBOX_PAD_FRAC times the shorter side of the box, applied
        on all four edges and clamped to the image dimensions.
        """
        if bbox is None:
            return image

        xmin, ymin, xmax, ymax = bbox
        w, h = image.size
        pad  = int(BBOX_PAD_FRAC * min(xmax - xmin, ymax - ymin))

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



# Public API
def build_datasets(
    data_root: str = DATA_ROOT,
    use_data_adapt: bool = False,
) -> Tuple["WildlifeDataset", "WildlifeDataset",
           "WildlifeDataset", Dict[str, int], Dict[int, str]]:
    """
    Scan voc_day and voc_night, build datasets according to the split strategy.

    Split ratios
    ------------
    voc_day   : 100% training (all records used).
    voc_night : 80% test, 20% validation (class-balanced, seeded).

    When use_data_adapt is True, a second copy of the full daytime training
    records is created with night-simulation transforms and concatenated with
    the original training set.  The combined dataset is returned as train_ds.
    The adapted transform is imported from data_adaptation.py at call time.

    Parameters
    ----------
    data_root      : path to the directory containing voc_day and voc_night
    use_data_adapt : whether to append night-simulated daytime images to training

    Returns
    -------
    train_ds     : WildlifeDataset (or ConcatDataset when adaptation is on)
    test_ds      : WildlifeDataset  80% of voc_night
    val_night_ds : WildlifeDataset  20% of voc_night
    label2idx    : dict mapping species name to integer index
    idx2label    : dict mapping integer index to species name
    """
    root = Path(data_root)

    day_records   = scan_voc_split(root / "voc_day")
    night_records = scan_voc_split(root / "voc_night")

    # Build vocabulary from voc_day only so the output head is fully defined
    # before any test data is seen.
    all_labels = sorted({label for _, label, _ in day_records})
    label2idx  = {lbl: i for i, lbl in enumerate(all_labels)}
    idx2label  = {i: lbl for lbl, i in label2idx.items()}

    print(f"\nClasses ({len(all_labels)}): {all_labels}\n")

    # voc_night: 80% test / 20% validation (class-balanced)
    night_known = [r for r in night_records if r[1] in label2idx]
    test_records, val_night_records = stratified_split(
        night_known, val_fraction=0.20, seed=SPLIT_SEED
    )

    print(f"  voc_day   : {len(day_records)} training samples (100%)")
    print(f"  voc_night : {len(test_records)} test / {len(val_night_records)} validation\n")

    # Original training dataset with standard augmentation
    original_train_ds = WildlifeDataset(day_records, label2idx, get_train_transforms())

    if use_data_adapt:
        # Adapted dataset: same records, night-simulation transform
        from data_adaptation import get_adapted_train_transforms
        adapted_train_ds = WildlifeDataset(
            day_records, label2idx, get_adapted_train_transforms()
        )
        # Concatenate original and adapted datasets so the model sees both
        train_ds = ConcatDataset([original_train_ds, adapted_train_ds])
        print(f"  Training set: {len(original_train_ds)} original + "
              f"{len(adapted_train_ds)} adapted = {len(train_ds)} total samples.")
    else:
        train_ds = original_train_ds
        print(f"  Training set: {len(train_ds)} samples (original only).")

    test_ds      = WildlifeDataset(test_records,      label2idx, get_eval_transforms())
    val_night_ds = WildlifeDataset(val_night_records, label2idx, get_eval_transforms())

    return train_ds, test_ds, val_night_ds, label2idx, idx2label


def get_dataloaders(
    train_ds:     Dataset,
    test_ds:      "WildlifeDataset",
    val_night_ds: "WildlifeDataset",
    batch_size:   int = 16,
    num_workers:  int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train, test, and val_night DataLoaders."""
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
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
    return train_loader, test_loader, val_night_loader



# Self-test
if __name__ == "__main__":
    import sys

    data_root = sys.argv[1] if len(sys.argv) > 1 else DATA_ROOT
    train_ds, test_ds, val_night_ds, label2idx, idx2label = \
        build_datasets(data_root, use_data_adapt=False)

    print(f"Train      : {len(train_ds)}")
    print(f"Test       : {len(test_ds)}")
    print(f"Val night  : {len(val_night_ds)}")
    print(f"Num classes: {len(label2idx)}")

    train_loader, _, _ = get_dataloaders(
        train_ds, test_ds, val_night_ds, batch_size=4, num_workers=0
    )
    imgs, labels = next(iter(train_loader))
    print(f"Batch shape : {imgs.shape}")
    print(f"Labels      : {[idx2label[i.item()] for i in labels]}")
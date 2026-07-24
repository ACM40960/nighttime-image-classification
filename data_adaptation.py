import random
from typing import Literal

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torchvision import transforms
import cv2

# Constants (shared with dataset.py — kept here to avoid circular imports)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_SIZE    = 224

# Individual transform classes
class RandomGreyscaleSimulation:
    """
    With probability `p`, convert image to greyscale and back to RGB.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            # Convert to greyscale then back to RGB so tensor shape is preserved
            img = img.convert("L").convert("RGB")
        return img


class CLAHETransform:
    """
    Apply CLAHE (Contrast-Limited Adaptive Histogram Equalisation) in LAB space.
    """

    def __init__(self, clip_limit: float = 8.0,
                 tile_grid: tuple = (8, 8), p: float = 0.7):
        self.clip_limit = clip_limit
        self.tile_grid  = tile_grid
        self.p          = p

    def _clahe_cv2(self, img: Image.Image) -> Image.Image:
        img_np  = np.array(img)
        lab     = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=self.clip_limit,
                                   tileGridSize=self.tile_grid)
        l       = clahe.apply(l)
        merged  = cv2.merge([l, a, b])
        result  = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        return self._clahe_cv2(img)


class RandomNightNoise:
    """
    Add Gaussian noise to a PIL image to simulate high-ISO sensor grain.
    """

    def __init__(self, std_range: tuple = (5, 25), p: float = 0.6):
        self.std_range = std_range
        self.p         = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        std    = random.uniform(*self.std_range)
        arr    = np.array(img, dtype=np.float32)
        noise  = np.random.normal(0, std, arr.shape).astype(np.float32)
        noisy  = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)


class RandomBrightnessSuppression:
    """
    Apply a random gamma curve to darken the image, simulating night exposure.
    """

    def __init__(self, gamma_range: tuple = (1.2, 2.2), p: float = 0.6):
        self.gamma_range = gamma_range
        self.p           = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        gamma = random.uniform(*self.gamma_range)
        return TF.adjust_gamma(img, gamma=gamma)


class RandomContrastCompression:
    """
    Reduce global contrast to simulate the flat-contrast look of night flash.
    """

    def __init__(self, factor_range: tuple = (0.3, 0.7), p: float = 0.5):
        self.factor_range = factor_range
        self.p            = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        factor = random.uniform(*self.factor_range)
        return TF.adjust_contrast(img, contrast_factor=factor)


class RandomFlashHotspot:
    """
    Overlay a faint bright ellipse near the image centre to simulate the
    characteristic overexposed hotspot produced by a night-flash camera trap.
    """

    def __init__(self, intensity_range: tuple = (0.05, 0.2), p: float = 0.3):
        self.intensity_range = intensity_range
        self.p               = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        arr     = np.array(img, dtype=np.float32) / 255.0
        h, w, _ = arr.shape
        cy, cx  = h // 2, w // 2

        # Build a 2D Gaussian weight mask centred on (cx, cy)
        sigma    = min(h, w) * 0.35
        ys, xs   = np.ogrid[:h, :w]
        mask     = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
        mask     = mask[:, :, np.newaxis]               # (H, W, 1)

        intensity = random.uniform(*self.intensity_range)
        arr       = np.clip(arr + intensity * mask, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))


 
# Composed pipelines
def get_adapted_train_transforms(
    strength: Literal["light", "medium", "strong"] = "medium",
) -> transforms.Compose:
    """
    Return a training transform pipeline with data-level domain adaptation.
    """

    #  Technique parameters by strength
    if strength == "light":
        greyscale_p    = 0.2
        clahe_p        = 0.3
        noise_p        = 0.3
        noise_std      = (3, 12)
        brightness_p   = 0.3
        gamma_range    = (1.0, 1.5)
        contrast_p     = 0.2
        contrast_range = (0.6, 0.9)
        hotspot_p      = 0.1

    elif strength == "strong":
        greyscale_p    = 0.8
        clahe_p        = 0.9
        noise_p        = 0.8
        noise_std      = (15, 40)
        brightness_p   = 0.8
        gamma_range    = (1.8, 3.0)
        contrast_p     = 0.7
        contrast_range = (0.2, 0.5)
        hotspot_p      = 0.5

    else:  # medium (default)
        greyscale_p    = 0.5
        clahe_p        = 0.7
        noise_p        = 0.6
        noise_std      = (5, 25)
        brightness_p   = 0.6
        gamma_range    = (1.2, 2.2)
        contrast_p     = 0.5
        contrast_range = (0.3, 0.7)
        hotspot_p      = 0.3

    return transforms.Compose([
        # Spatial augmentations
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),

        # Night simulation — data-level domain adaptation 
        # Step 1: Greyscale simulation (near-IR desaturation)
        RandomGreyscaleSimulation(p=greyscale_p),

        # Step 2: CLAHE in LAB space
        CLAHETransform(clip_limit=8.0, tile_grid=(8, 8), p=clahe_p),

        # Step 3: Gamma darkening (low-light brightness suppression)
        RandomBrightnessSuppression(gamma_range=gamma_range, p=brightness_p),

        # Step 4: Contrast compression (flat dynamic range of night flash)
        RandomContrastCompression(factor_range=contrast_range, p=contrast_p),

        # Step 5: Sensor noise (high-ISO grain)
        RandomNightNoise(std_range=noise_std, p=noise_p),

        # Step 6: Flash hotspot overlay (bright centre, dark surroundings)
        RandomFlashHotspot(intensity_range=(0.05, 0.2), p=hotspot_p),

        # Tensor conversion and normalisation
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_night_test_transforms() -> transforms.Compose:
    """
    Test-time transform for nighttime images.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


 
# Quick visual sanity check
 

if __name__ == "__main__":
    """
    Load a single image and save the adapted version alongside the original
    """
    import sys
    from pathlib import Path

    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    strength = sys.argv[2] if len(sys.argv) > 2 else "medium"

    if img_path is None or not img_path.exists():
        print("Usage: python data_adaptation.py <image_path> [light|medium|strong]")
        sys.exit(1)

    img = Image.open(img_path).convert("RGB")

    # Apply each transform individually so we can save intermediate results
    transform = get_adapted_train_transforms(strength=strength)

    # Save original
    out_dir = Path("adaptation_preview")
    out_dir.mkdir(exist_ok=True)
    img.save(out_dir / "original.jpg")

    # Apply and save adapted (de-normalise for viewing)
    tensor = transform(img)                      # (3, H, W) normalised
    mean   = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std    = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    vis    = (tensor * std + mean).clamp(0, 1)
    vis_pil = transforms.ToPILImage()(vis)
    out_path = out_dir / f"adapted_{strength}.jpg"
    vis_pil.save(out_path)

    print(f"Original  : {out_dir}/original.jpg")
    print(f"Adapted   : {out_path}")
    print(f"Strength  : {strength}")
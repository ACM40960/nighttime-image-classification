import random

import cv2
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_SIZE    = 224

# Individual transform classes
class RandomGreyscaleSimulation:
    """
    Convert the image to greyscale and back to RGB with probability p.

    Simulates near-infrared desaturation common in night camera traps where
    the sensor loses colour information under low-light conditions.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            img = img.convert("L").convert("RGB")
        return img


class CLAHETransform:
    """
    Apply Contrast-Limited Adaptive Histogram Equalisation in LAB colour space.

    CLAHE is applied only to the luminance channel (L), leaving the colour
    channels (A, B) unchanged to avoid colour-shift artefacts.

    Parameters
    ----------
    clip_limit : float
        Maximum slope of the cumulative distribution function mapping.
    tile_grid  : tuple
        Number of tiles in (rows, cols) for adaptive local computation.
    p          : float
        Probability of applying this transform.
    """

    def __init__(self, clip_limit: float = 8.0,
                 tile_grid: tuple = (8, 8), p: float = 0.7):
        self.clip_limit = clip_limit
        self.tile_grid  = tile_grid
        self.p          = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        img_np  = np.array(img)
        lab     = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe   = cv2.createCLAHE(clipLimit=self.clip_limit,
                                   tileGridSize=self.tile_grid)
        l       = clahe.apply(l)
        merged  = cv2.merge([l, a, b])
        result  = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(result)


class RandomNightNoise:
    """
    Add Gaussian noise to simulate high-ISO sensor grain in night captures.

    Parameters
    ----------
    std_range : tuple
        (min_std, max_std) of the Gaussian noise in the 0-255 pixel scale.
    p         : float
        Probability of applying this transform.
    """

    def __init__(self, std_range: tuple = (5, 25), p: float = 0.6):
        self.std_range = std_range
        self.p         = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        std   = random.uniform(*self.std_range)
        arr   = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, std, arr.shape).astype(np.float32)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)


class RandomBrightnessSuppression:
    """
    Apply a random gamma curve to darken the image, simulating low-light exposure.

    A gamma value greater than 1 darkens the image.  The value is sampled
    uniformly from gamma_range to add stochasticity across training epochs.

    Parameters
    ----------
    gamma_range : tuple
        (min_gamma, max_gamma).
    p           : float
        Probability of applying this transform.
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
    Reduce global image contrast to simulate the flat dynamic range produced
    by night-flash illumination.

    Parameters
    ----------
    factor_range : tuple
        (min_factor, max_factor).  A factor of 0 produces a uniform grey
        image; a factor of 1 leaves the image unchanged.
    p            : float
        Probability of applying this transform.
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
    Overlay a Gaussian-weighted brightness increase near the image centre to
    simulate the overexposed hotspot produced by a night-flash camera trap.

    Parameters
    ----------
    intensity_range : tuple
        (min, max) brightness delta added at the hotspot centre (0-1 scale).
    p               : float
        Probability of applying this transform.
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

        sigma = min(h, w) * 0.35
        ys, xs = np.ogrid[:h, :w]
        mask   = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
        mask   = mask[:, :, np.newaxis]

        intensity = random.uniform(*self.intensity_range)
        arr       = np.clip(arr + intensity * mask, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))


# Composed pipeline
def get_adapted_train_transforms() -> transforms.Compose:
    """
    Return the night-simulation training transform pipeline (medium strength).

    The pipeline applies spatial augmentations first, followed by the
    night-simulation techniques in order of global-to-local effect, then
    converts to a normalised tensor.

    Returns
    -------
    transforms.Compose
        Drop-in replacement for dataset.get_train_transforms().
    """
    return transforms.Compose([
        # Spatial augmentations (applied before pixel-level changes to avoid
        # interpolation artefacts on noisy images).
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),

        # Night simulation techniques (global to local).
        RandomGreyscaleSimulation(p=0.5),
        CLAHETransform(clip_limit=8.0, tile_grid=(8, 8), p=0.7),
        RandomBrightnessSuppression(gamma_range=(1.2, 2.2), p=0.6),
        RandomContrastCompression(factor_range=(0.3, 0.7), p=0.5),
        RandomNightNoise(std_range=(5, 25), p=0.6),
        RandomFlashHotspot(intensity_range=(0.05, 0.2), p=0.3),

        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
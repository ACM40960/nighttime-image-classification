import argparse
import io
import sys
from pathlib import Path
from typing import List

# Add project root to path so train.py can be imported from app/app.py.
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

# Inference transform
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_SIZE    = 224


def get_inference_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# Model loading
def load_model(checkpoint_path: str, device: torch.device):
    """
    Load Dinov2Classifier from a checkpoint saved by train.py.

    Returns (model, label2idx, idx2label).
    """
    from train import Dinov2Classifier

    ckpt        = torch.load(checkpoint_path, map_location=device)
    label2idx   = ckpt["label2idx"]
    idx2label   = {int(k): v for k, v in ckpt["idx2label"].items()}
    num_classes = ckpt["num_classes"]
    use_supcon  = ckpt.get("use_supcon", False)

    model = Dinov2Classifier(
        num_classes=num_classes,
        dropout=0.3,
        use_supcon=use_supcon,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Checkpoint loaded: epoch {ckpt['epoch']}, {num_classes} classes.")
    print(f"Classes: {sorted(label2idx.keys())}")

    return model, label2idx, idx2label


# Application state (set once at startup)
_model     = None
_label2idx = None
_idx2label = None
_device    = None
_transform = get_inference_transform()

# HTML frontend is read once at import time and served from memory.
INDEX_HTML_PATH = Path(__file__).parent / "index.html"
INDEX_HTML      = INDEX_HTML_PATH.read_text(encoding="utf-8")


# FastAPI app
app = FastAPI(
    title       = "Wildlife Camera-Trap Species Classifier",
    description = "DINOv2-base fine-tuned for day-to-night species identification.",
    version     = "1.0.0",
)


# Response schema
class PredictionItem(BaseModel):
    species:    str
    confidence: float

class PredictionResponse(BaseModel):
    species:    str
    confidence: float
    top5:       List[PredictionItem]


# Inference
@torch.no_grad()
def run_inference(image: Image.Image) -> PredictionResponse:
    """
    Run a single image through the model and return the top-5 predictions.

    The full image is resized to 224x224 and classified without any crop.
    """
    tensor = _transform(image).unsqueeze(0).to(_device)   # (1, 3, 224, 224)
    logits = _model(tensor, return_projection=False)       # (1, num_classes)
    probs  = F.softmax(logits, dim=1).squeeze(0)           # (num_classes,)

    top5_probs, top5_idx = probs.topk(min(5, len(_idx2label)))
    top5 = [
        PredictionItem(
            species    = _idx2label[idx.item()],
            confidence = round(prob.item(), 4),
        )
        for prob, idx in zip(top5_probs, top5_idx)
    ]

    return PredictionResponse(
        species    = top5[0].species,
        confidence = top5[0].confidence,
        top5       = top5,
    )


# Endpoints
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    """
    Return model metadata consumed by the frontend to populate dynamic labels.
    Called once by index.html on page load via fetch("/info").
    """
    if _label2idx is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "num_classes": len(_label2idx),
        "classes":     sorted(_label2idx.keys()),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accept an image file upload and return the predicted species.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. Upload a valid JPEG or PNG file.",
        )

    return run_inference(image)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the static HTML frontend from app/index.html."""
    return INDEX_HTML


# Entry point
def parse_args():
    p = argparse.ArgumentParser(
        description="Wildlife classifier inference server."
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to best_model_{ts}.pt saved by train.py."
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {_device}")

    _model, _label2idx, _idx2label = load_model(args.checkpoint, _device)

    uvicorn.run(app, host=args.host, port=args.port)
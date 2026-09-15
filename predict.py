"""
SignalScope — predict interface (Section 4.1 of the brief).

Loads the trained checkpoint once, then exposes:

    predict_image(pil_image) -> {
        "label": "real" | "ai_generated",
        "confidence": float,          # 0-1, calibrated probability of the winning class
        "raw_score": float            # 0-1, raw sigmoid output ("probability of AI-generated")
    }

Used directly by app/server.py. Can also be run standalone:

    python predict.py path/to/image.jpg
"""

import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

CHECKPOINT_PATH = "model/weights/signalscope.pt"
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_img_size = 224
_idx_to_label = {0: "real", 1: "ai_generated"}  # overwritten from checkpoint if present


def _build_model():
    m = models.efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, 1)
    return m


def load_model(checkpoint_path=CHECKPOINT_PATH):
    global _model, _img_size, _idx_to_label
    ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=False)
    _model = _build_model()
    _model.load_state_dict(ckpt["model_state"])
    _model.to(_device)
    _model.eval()
    _img_size = ckpt.get("img_size", 224)

    class_to_idx = ckpt.get("class_to_idx")
    if class_to_idx:
        # class_to_idx looks like {"fake": 1, "real": 0} or similar; invert it
        _idx_to_label = {v: ("ai_generated" if k.lower() in ("fake", "ai", "ai_generated") else "real")
                          for k, v in class_to_idx.items()}
    return _model


def _preprocess(pil_image):
    tf = transforms.Compose([
        transforms.Resize((_img_size, _img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(pil_image.convert("RGB")).unsqueeze(0)


def predict_image(pil_image):
    if _model is None:
        load_model()

    x = _preprocess(pil_image).to(_device)
    with torch.no_grad():
        logit = _model(x).squeeze(1)
        raw_score = torch.sigmoid(logit).item()  # P(class index 1)

    label = _idx_to_label.get(1, "ai_generated") if raw_score >= 0.5 else _idx_to_label.get(0, "real")
    confidence = raw_score if raw_score >= 0.5 else 1 - raw_score

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "raw_score": round(raw_score, 4),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)

    img = Image.open(sys.argv[1])
    result = predict_image(img)
    print(result)

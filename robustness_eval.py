"""
SignalScope — robustness-to-degradation report (Bonus C).

Takes the trained checkpoint and the validation set, and measures how
accuracy holds up as images are re-compressed, resized down and back up,
or blurred — the conditions a real detector actually meets in the wild
(screenshots, re-uploads, social-media recompression).

Run (after model/train.py has produced a checkpoint):

    python model/robustness_eval.py --data_dir data

Writes report/robustness_metrics.json with AUC per degradation, for the
"Robustness to degradation" row in the UI and README.
"""

import argparse
import io
import json
import os

import torch
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score
from torchvision import datasets

import predict

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MODEL_DIR)


def _degrade(pil_image, kind):
    img = pil_image.convert("RGB")
    if kind == "clean":
        return img
    if kind == "jpeg_q30":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=30)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if kind == "downscale_2x":
        w, h = img.size
        small = img.resize((max(1, w // 2), max(1, h // 2)), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    if kind == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=2))
    raise ValueError(kind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--checkpoint", default=os.path.join(_MODEL_DIR, "weights", "signalscope.pt"))
    parser.add_argument("--max_images", type=int, default=300,
                         help="Cap for speed — this is a spot-check, not a full re-evaluation.")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print("No trained checkpoint found at", args.checkpoint)
        print("Run model/train.py first.")
        return

    predict.load_model(args.checkpoint)

    val_dir = args.data_dir if os.path.isabs(args.data_dir) else os.path.join(_PROJECT_ROOT, args.data_dir)
    val_ds = datasets.ImageFolder(os.path.join(val_dir, "val"))
    samples = val_ds.samples[: args.max_images]

    degradations = ["clean", "jpeg_q30", "downscale_2x", "gaussian_blur"]
    results = {}

    for kind in degradations:
        labels, scores = [], []
        for path, label in samples:
            img = Image.open(path)
            degraded = _degrade(img, kind)
            r = predict.predict_image(degraded)
            labels.append(label)
            scores.append(r["raw_score"])
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = None
        results[kind] = auc
        print(f"{kind:15s} AUC={auc}")

    report_dir = os.path.join(_PROJECT_ROOT, "report")
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "robustness_metrics.json"), "w") as f:
        json.dump({"n_images": len(samples), "auc_by_degradation": results}, f, indent=2)

    print("Done. Wrote report/robustness_metrics.json")


if __name__ == "__main__":
    main()

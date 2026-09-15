"""
SignalScope — Core task training script
Real vs. AI-generated image classification via transfer learning.

Expected data layout (ImageFolder format):

    data/
      train/
        real/   ...jpg
        fake/   ...jpg
      val/
        real/
        fake/
      test/                 <-- optional local held-out sample for your own checks
        real/
        fake/

Run:
    pip install -r requirements.txt
    python train.py --data_dir data --epochs 8 --out model/weights/signalscope.pt

This uses an EfficientNet-B0 backbone pretrained on ImageNet, with a fresh
binary classification head — "transfer learning encouraged" per the brief.
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix


def build_model(device):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)  # single logit -> sigmoid
    return model.to(device)


def get_loaders(data_dir, img_size, batch_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        # Randomly re-compress / blur sometimes, so the model doesn't only
        # learn clean-image artefacts (helps with the robustness bonus too).
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=eval_tf)

    # ImageFolder assigns class indices alphabetically -> confirm "fake"=1, "real"=0
    print("Class mapping:", train_ds.class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, train_ds.class_to_idx


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y.numpy().tolist())

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    auc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, preds, average="macro")
    cm = confusion_matrix(all_labels, preds).tolist()
    return {"auc": auc, "macro_f1": f1, "confusion_matrix": cm}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--out", default="model/weights/signalscope.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader, class_to_idx = get_loaders(args.data_dir, args.img_size, args.batch_size)
    model = build_model(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        start = time.time()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device)
            optimizer.zero_grad()
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch+1}/{args.epochs} "
              f"loss={running_loss/len(train_loader.dataset):.4f} "
              f"val_auc={metrics['auc']:.4f} val_f1={metrics['macro_f1']:.4f} "
              f"({time.time()-start:.1f}s)")

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": class_to_idx,
                "img_size": args.img_size,
                "metrics": metrics,
            }, args.out)
            print(f"  -> saved new best checkpoint (AUC {best_auc:.4f})")

    # Write a metrics summary for the /report folder
    with open("report/train_metrics.json", "w") as f:
        json.dump({"best_val_auc": best_auc}, f, indent=2)

    print("Done. Best checkpoint:", args.out)


if __name__ == "__main__":
    main()

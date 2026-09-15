# SignalScope — Telling Real From Synthetic

Real vs. AI-generated image classifier with a live web demo.

## 1. What's built

- **Core task**: binary real / AI-generated classifier (EfficientNet-B0 transfer learning).
- **Bonus A (explanation)**: cue list + heatmap panel in the UI (static preview for now — see `report/` for how to wire in Grad-CAM).
- **Bonus F (deployable interface)**: drag-and-drop web app, Flask backend, live prediction.

## 2. Project structure

```
signalscope/
  app/
    server.py         Flask app: serves the UI + /api/predict
    static/
      index.html       Frontend (HTML/CSS/JS, single file)
  model/
    train.py           Training script (transfer learning)
    predict.py         Inference helper used by the server
    weights/
      signalscope.pt   Saved checkpoint (created after training)
  report/
    train_metrics.json  Auto-written after training
  requirements.txt
  README.md
```

## 3. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Get the dataset

Use the provided CIFAKE-style real-vs-synthetic dataset (kickoff dataset), or the
public CIFAKE dataset on Kaggle as a stand-in while building. Arrange it as:

```
data/
  train/
    real/   ...images
    fake/   ...images
  val/
    real/
    fake/
```

(You may add other public synthetic-image datasets, e.g. GenImage, to `train/` —
cite them in this README per the submission rules. Never train on the organizers'
held-out test set.)

## 5. Train

```bash
python model/train.py --data_dir data --epochs 8
```

This saves the best checkpoint to `model/weights/signalscope.pt` and writes
validation AUC / macro-F1 / confusion matrix to `report/train_metrics.json`.

## 6. Run the app

```bash
python app/server.py
```

Open **http://localhost:5000** — drop an image in and it will call the trained
model and show a live verdict + confidence.

## 7. Reported metrics (fill in after training)

| Metric | Overall | Unseen-generator split |
|---|---|---|
| ROC-AUC | — | — |
| Macro-F1 | — | — |
| Accuracy @ threshold | — | — |
| FPR @ threshold | — | — |

## 8. Known limitations

- No GPU training loop optimisation beyond a standard AdamW + cosine schedule.
- Generator attribution (Bonus B), provenance/metadata (Bonus D), and
  multimodal consistency (Bonus E) are not implemented yet.
- Explanation panel currently shows illustrative cues; wiring real Grad-CAM
  output into the same UI is the next step (see inline comments in
  `app/static/index.html`, section `#explain`).

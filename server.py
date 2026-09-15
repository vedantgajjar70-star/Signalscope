"""
SignalScope — backend server.

Serves the frontend (app/static/index.html) and a /api/predict endpoint that
runs the trained model on an uploaded image.

Run:
    pip install -r requirements.txt
    python app/server.py

Then open http://localhost:5000 in your browser.
"""

import os
import sys

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

# Make "model/" importable regardless of where this script is launched from.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))
import predict  # noqa: E402  (model/predict.py)

app = Flask(__name__, static_folder="static", static_url_path="")

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "weights", "signalscope.pt")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Send it as form field 'image'."}), 400

    file = request.files["image"]
    try:
        img = Image.open(file.stream)
    except Exception:
        return jsonify({"error": "Could not read the uploaded file as an image."}), 400

    try:
        result = predict.predict_image(img)
    except FileNotFoundError:
        return jsonify({
            "error": "No trained model found. Run model/train.py first, "
                     "or point CHECKPOINT_PATH to your weights file."
        }), 500

    return jsonify(result)


if __name__ == "__main__":
    # Load the model once at startup instead of on the first request.
    if os.path.exists(CHECKPOINT_PATH):
        predict.load_model(CHECKPOINT_PATH)
        print("Model loaded from", CHECKPOINT_PATH)
    else:
        print("WARNING: no checkpoint found at", CHECKPOINT_PATH)
        print("The server will start, but /api/predict will fail until you train a model.")

    app.run(host="0.0.0.0", port=5000, debug=True)

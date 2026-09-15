"""
SignalScope — training-free forensic detector.

Used automatically by predict.py whenever no trained checkpoint exists yet
(fresh clone, before anyone runs model/train.py) and, always, as the source
of the human-readable "cues" shown in the explanation panel.

This is NOT a substitute for the trained EfficientNet-B0 model in
model/train.py — it's an honest fallback so the demo gives a real,
non-fabricated answer on any photo from the moment the repo is downloaded.
It combines five classical image-forensics signals, each individually
weak but genuinely informative:

  1. Noise-residual energy  — real camera sensors leave high-frequency
                               shot/read noise; heavy smoothing during
                               generation/upsampling tends to suppress it.
  2. Spectral irregularity  — natural photos roughly follow a smooth 1/f
                               radial power-spectrum falloff; GAN/diffusion
                               upsampling can leave grid-like energy that
                               breaks this falloff.
  3. Edge-sharpness variance — real photos have depth-of-field / focus
                               falloff, so sharpness varies block to block;
                               unnaturally uniform sharpness is a mild tell.
  4. Colour/saturation spread — some generators produce an unnaturally
                               narrow saturation distribution.
  5. EXIF camera metadata    — presence of Make/Model tags is fair (not
                               certain) evidence of a real camera photo;
                               only applied to formats that normally carry
                               EXIF (JPEG/TIFF), and weighted modestly since
                               real photos are routinely stripped of it by
                               messaging apps and social platforms.

Expect this heuristic to land somewhere around the 65-75% accuracy range
typical of classical forensic detectors in the literature — clearly useful,
clearly not as strong as a trained deep model on in-distribution data.
Once model/train.py produces model/weights/signalscope.pt, predict.py
automatically switches to the trained model for the verdict and uses real
Grad-CAM for the heatmap instead of this module's noise-residual map.
"""

import base64
import io

import numpy as np
from PIL import Image, ImageFilter

AMBER = np.array([242, 169, 59], dtype=np.float64)
DARK = np.array([16, 21, 28], dtype=np.float64)


def _clip01(v):
    return max(0.0, min(1.0, float(v)))


def _prep(pil_image, max_side=512):
    img = pil_image.convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return img


def _noise_signal(gray_img):
    gray = np.asarray(gray_img, dtype=np.float64)
    blurred = np.asarray(gray_img.filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float64)
    residual = gray - blurred
    noise_std = float(np.std(residual))
    # Calibration note: these thresholds are heuristic, chosen from typical
    # 8-bit sensor-noise magnitudes, not fit on a labelled dataset.
    suspicion = _clip01(1.0 - (noise_std - 1.5) / 5.5)
    return suspicion, noise_std, residual


def _spectral_signal(gray_img):
    gray = np.asarray(gray_img, dtype=np.float64)
    h, w = gray.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    spectrum = np.fft.fftshift(np.fft.fft2(gray * window))
    mag = np.log1p(np.abs(spectrum))

    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    max_r = min(cy, cx)
    if max_r < 8:
        return 0.4, 0.0

    radial = np.zeros(max_r)
    counts = np.bincount(r.ravel(), minlength=max_r + 1)[:max_r]
    sums = np.bincount(r.ravel(), weights=mag.ravel(), minlength=max_r + 1)[:max_r]
    nonzero = counts > 0
    radial[nonzero] = sums[nonzero] / counts[nonzero]

    lo, hi = int(max_r * 0.15), int(max_r * 0.85)
    if hi - lo < 5:
        return 0.4, 0.0
    xs = np.arange(lo, hi)
    ys = radial[lo:hi]
    coeffs = np.polyfit(xs, ys, 1)
    fit = np.polyval(coeffs, xs)
    residual_var = float(np.mean((ys - fit) ** 2))
    suspicion = _clip01(residual_var / 0.06)
    return suspicion, residual_var


def _edge_regularity_signal(gray_img):
    gray = np.asarray(gray_img, dtype=np.float64)
    h, w = gray.shape
    bs = max(16, min(h, w) // 12)
    scores = []
    for y0 in range(0, h - bs, bs):
        for x0 in range(0, w - bs, bs):
            block = gray[y0:y0 + bs, x0:x0 + bs]
            gy, gx = np.gradient(block)
            scores.append(float(np.var(gx) + np.var(gy)))
    if len(scores) < 4:
        return 0.3, 0.0
    scores = np.array(scores)
    cv = float(np.std(scores) / (np.mean(scores) + 1e-6))
    suspicion = _clip01(1.0 - cv / 1.2)
    return suspicion, cv


def _color_signal(rgb_img):
    hsv = np.asarray(rgb_img.convert("HSV"), dtype=np.float64)
    sat = hsv[..., 1] / 255.0
    std_sat = float(np.std(sat))
    suspicion = _clip01(1.0 - std_sat / 0.28)
    return suspicion, std_sat


def _exif_signal(pil_image):
    applicable = pil_image.format in ("JPEG", "MPO", "TIFF")
    if not applicable:
        return None, False
    try:
        exif = pil_image.getexif()
    except Exception:
        exif = None
    has_camera_tags = False
    if exif:
        make = exif.get(271)   # Make
        model = exif.get(272)  # Model
        has_camera_tags = bool(make or model)
    suspicion = 0.15 if has_camera_tags else 0.65
    return suspicion, has_camera_tags


def render_heatmap(heat_lowres, base_rgb_img, out_size):
    """heat_lowres: 2D array in any range (will be min-max normalised).
    base_rgb_img: PIL RGB image to overlay onto.
    out_size: (w, h) of the final composited PNG.
    Returns base64-encoded PNG bytes (no data-uri prefix).
    """
    heat = heat_lowres.astype(np.float64)
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()
    heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(out_size, Image.BICUBIC)
    heat_px = np.asarray(heat_img, dtype=np.float64) / 255.0

    color = (DARK[None, None, :] + (AMBER - DARK)[None, None, :] * heat_px[..., None]).astype(np.uint8)
    alpha = (50 + heat_px * 160).astype(np.uint8)
    overlay = Image.fromarray(np.dstack([color, alpha]), mode="RGBA")

    base = base_rgb_img.convert("RGBA").resize(out_size)
    composite = Image.alpha_composite(base, overlay).convert("RGB")

    buf = io.BytesIO()
    composite.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _block_downsample(arr, blocks=16):
    h, w = arr.shape
    bs_h, bs_w = max(1, h // blocks), max(1, w // blocks)
    hh, ww = (h // bs_h) * bs_h, (w // bs_w) * bs_w
    cropped = arr[:hh, :ww]
    return cropped.reshape(hh // bs_h, bs_h, ww // bs_w, bs_w).mean(axis=(1, 3))


def analyze(pil_image):
    work = _prep(pil_image)
    gray_img = work.convert("L")

    noise_susp, noise_std, residual = _noise_signal(gray_img)
    spec_susp, spec_var = _spectral_signal(gray_img)
    edge_susp, edge_cv = _edge_regularity_signal(gray_img)
    color_susp, sat_std = _color_signal(work)
    exif_susp, has_camera_tags = _exif_signal(pil_image)

    weights = {"noise": 0.30, "spectral": 0.28, "edge": 0.14, "color": 0.10, "exif": 0.18}
    signals = {"noise": noise_susp, "spectral": spec_susp, "edge": edge_susp, "color": color_susp}
    if exif_susp is not None:
        signals["exif"] = exif_susp

    active_weight = sum(weights[k] for k in signals)
    score = sum(signals[k] * weights[k] for k in signals) / active_weight
    score = _clip01(score)

    cues = [
        {
            "name": "Sensor-noise residual",
            "detail": ("High-frequency residual is unusually low (std {:.2f}) — a smoothing tell."
                       if noise_susp > 0.55 else
                       "High-frequency residual (std {:.2f}) is consistent with a natural sensor.").format(noise_std),
            "score": round(noise_susp, 2),
        },
        {
            "name": "Frequency-spectrum shape",
            "detail": ("Radial spectrum deviates from the natural 1/f falloff (var {:.4f}) — possible generator artefact."
                       if spec_susp > 0.55 else
                       "Radial spectrum follows a smooth, camera-like falloff (var {:.4f}).").format(spec_var),
            "score": round(spec_susp, 2),
        },
        {
            "name": "Edge-sharpness variation",
            "detail": ("Sharpness is unusually uniform across the frame ({:.2f}x variation) for a real photo."
                       if edge_susp > 0.55 else
                       "Sharpness varies {:.2f}x across the frame — natural depth-of-field falloff.").format(edge_cv),
            "score": round(edge_susp, 2),
        },
        {
            "name": "Colour / saturation spread",
            "detail": ("Saturation spread (std {:.3f}) is narrower than typical camera output."
                       if color_susp > 0.55 else
                       "Saturation spread (std {:.3f}) matches a typical natural range.").format(sat_std),
            "score": round(color_susp, 2),
        },
    ]
    if exif_susp is not None:
        cues.append({
            "name": "Camera metadata (EXIF)",
            "detail": ("Camera make/model tags are present — evidence of a real camera capture."
                       if has_camera_tags else
                       "No camera make/model tags found — common for AI output, but also for "
                       "edited, screenshotted, or re-saved real photos."),
            "score": round(exif_susp, 2),
        })

    likely_family = None
    if score >= 0.5:
        if spec_susp > noise_susp + 0.15:
            likely_family = "diffusion-style upsampling pattern (rough guess, not a validated attribution model)"
        elif noise_susp > spec_susp + 0.15:
            likely_family = "GAN-style over-smoothing pattern (rough guess, not a validated attribution model)"
        else:
            likely_family = "generator family unclear from these signals"

    heat_blocks = _block_downsample(np.abs(residual), blocks=16)
    heatmap_b64 = render_heatmap(heat_blocks, work, work.size)

    return {
        "score": score,
        "cues": cues,
        "heatmap_b64": heatmap_b64,
        "likely_family": likely_family,
    }

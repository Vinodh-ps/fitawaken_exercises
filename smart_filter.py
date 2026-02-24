"""
Workout GIF → Smooth Mannequin Transformer
============================================
Actually destroys the original pencil/sketch texture and replaces it
with smooth, clean mannequin-like surfaces.

The key techniques:
1. Heavy bilateral + median filtering to KILL line-art texture
2. Edge-preserving smoothing that keeps silhouette but removes hatching
3. Rebuild surface shading using smooth gradients
4. Muscle areas get a clean colored surface (not just hue-shifted)

Requirements:
    pip install opencv-python-headless numpy Pillow

Usage:
    1. Put source GIFs in input_gifs/
    2. python transform_mannequin.py
    3. Output in output_gifs/
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageSequence

INPUT_DIR = "input_gifs"
OUTPUT_DIR = "output_gifs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── STYLE CONFIG ────────────────────────────────────────────────────

# Muscle highlight color (RGB)
MUSCLE_COLOR_LIGHT = np.array([80, 230, 255])    # Bright cyan
MUSCLE_COLOR_DARK = np.array([20, 100, 130])     # Dark cyan shadow

# Body mannequin color (RGB) — clean matte grey
BODY_COLOR_LIGHT = np.array([210, 215, 220])     # Highlight
BODY_COLOR_MID = np.array([150, 155, 162])        # Midtone
BODY_COLOR_DARK = np.array([65, 70, 78])          # Shadow

# Background
BG_COLOR = np.array([245, 245, 248])              # Clean near-white
# For dark bg, use: np.array([25, 27, 32])

# Smoothing intensity (higher = smoother = more mannequin-like)
SMOOTH_PASSES = 4          # Number of bilateral filter passes
SMOOTH_D = 9               # Bilateral filter diameter
SMOOTH_SIGMA_COLOR = 95    # Color sensitivity
SMOOTH_SIGMA_SPACE = 95    # Spatial sensitivity
MEDIAN_K = 7               # Median filter kernel (kills fine lines)


def process_frame(frame_pil):
    img_rgb = np.array(frame_pil.convert("RGB"))
    h, w = img_rgb.shape[:2]
    img_bgr = img_rgb[:, :, ::-1].copy()
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Detect background and create soft figure alpha
    # ═══════════════════════════════════════════════════════════════
    margin = 8
    edge_samples = np.concatenate([
        gray[0:margin, :].flatten(),
        gray[h-margin:h, :].flatten(),
        gray[:, 0:margin].flatten(),
        gray[:, w-margin:w].flatten()
    ])
    bg_mean = np.mean(edge_samples)
    bg_std = max(np.std(edge_samples), 5)
    is_dark_bg = bg_mean < 100

    if is_dark_bg:
        figure_alpha = np.clip((gray - bg_mean - bg_std * 1.5) / 45.0, 0, 1)
    else:
        figure_alpha = np.clip((bg_mean - bg_std * 1.5 - gray) / 60.0, 0, 1)

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Detect muscle regions
    # ═══════════════════════════════════════════════════════════════
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    is_red1 = (hue <= 15) & (sat >= 50) & (val >= 50)
    is_red2 = (hue >= 158) & (sat >= 50) & (val >= 50)
    is_orange = (hue >= 5) & (hue <= 30) & (sat >= 60) & (val >= 60)
    muscle_mask = (is_red1 | is_red2 | is_orange).astype(np.uint8) * 255

    # Clean and smooth muscle mask
    k_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    muscle_mask = cv2.morphologyEx(muscle_mask, cv2.MORPH_CLOSE, k_morph, iterations=2)
    muscle_mask = cv2.morphologyEx(muscle_mask, cv2.MORPH_OPEN, k_morph, iterations=1)
    muscle_float = cv2.GaussianBlur(muscle_mask.astype(np.float32) / 255.0, (9, 9), 0)
    muscle_float = np.clip(muscle_float, 0, 1)

    # Include muscles in figure alpha
    figure_alpha = np.maximum(figure_alpha, muscle_float)
    figure_alpha = cv2.GaussianBlur(figure_alpha, (7, 7), 0)
    figure_alpha = np.clip(figure_alpha, 0, 1)

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: KILL THE LINE-ART TEXTURE
    # ═══════════════════════════════════════════════════════════════
    # This is the critical step. We aggressively smooth the grayscale
    # to destroy pencil strokes, hatching, and line-art detail while
    # keeping the overall form/silhouette.

    smooth = gray.astype(np.uint8)

    # Median filter — excellent at removing thin lines while keeping edges
    smooth = cv2.medianBlur(smooth, MEDIAN_K)

    # Multiple bilateral filter passes — smooths surface while keeping silhouette
    for _ in range(SMOOTH_PASSES):
        smooth = cv2.bilateralFilter(smooth, SMOOTH_D, SMOOTH_SIGMA_COLOR, SMOOTH_SIGMA_SPACE)

    # One more median pass to catch any remaining fine texture
    smooth = cv2.medianBlur(smooth, 5)

    # Final light Gaussian to make it silky
    smooth = cv2.GaussianBlur(smooth, (5, 5), 0)

    smooth = smooth.astype(np.float32)

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: Normalize the smoothed luminance into a clean 0-1 range
    # ═══════════════════════════════════════════════════════════════
    fg_pixels = smooth[figure_alpha > 0.3]
    if len(fg_pixels) > 100:
        p2 = np.percentile(fg_pixels, 3)
        p98 = np.percentile(fg_pixels, 97)
    else:
        p2, p98 = 30, 220

    if p98 - p2 < 15:
        p98 = p2 + 15

    t = np.clip((smooth - p2) / (p98 - p2), 0, 1)

    # For dark backgrounds, bright pixels in original = lit areas on figure
    # For light backgrounds, dark pixels = the drawn figure (invert)
    if not is_dark_bg:
        t = 1.0 - t

    # ═══════════════════════════════════════════════════════════════
    # STEP 5: Build the mannequin body surface
    # ═══════════════════════════════════════════════════════════════
    # Map the smooth normalized luminance to a 3-stop color ramp:
    # dark → mid → light, creating a clean mannequin material

    body = np.zeros((h, w, 3), dtype=np.float32)

    # Two-segment ramp: [0..0.5] = dark→mid, [0.5..1] = mid→light
    t_low = np.clip(t * 2.0, 0, 1)         # 0-0.5 mapped to 0-1
    t_high = np.clip(t * 2.0 - 1.0, 0, 1)  # 0.5-1 mapped to 0-1
    is_lower = (t <= 0.5)

    for c in range(3):
        dark_val = BODY_COLOR_DARK[c]
        mid_val = BODY_COLOR_MID[c]
        light_val = BODY_COLOR_LIGHT[c]

        lower_blend = dark_val + (mid_val - dark_val) * t_low
        upper_blend = mid_val + (light_val - mid_val) * t_high

        body[:, :, c] = np.where(is_lower, lower_blend, upper_blend)

    # ═══════════════════════════════════════════════════════════════
    # STEP 6: Build the muscle surface
    # ═══════════════════════════════════════════════════════════════
    # Same idea — smooth luminance drives a color ramp, but in muscle color

    muscle_surface = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        muscle_surface[:, :, c] = MUSCLE_COLOR_DARK[c] + \
            (MUSCLE_COLOR_LIGHT[c] - MUSCLE_COLOR_DARK[c]) * t

    # ═══════════════════════════════════════════════════════════════
    # STEP 7: Blend muscles onto body
    # ═══════════════════════════════════════════════════════════════
    m3 = np.dstack([muscle_float] * 3)
    figure_surface = body * (1.0 - m3) + muscle_surface * m3

    # ═══════════════════════════════════════════════════════════════
    # STEP 8: Add specular highlights (shiny mannequin look)
    # ═══════════════════════════════════════════════════════════════
    if np.any(figure_alpha > 0.3):
        spec_thresh = np.percentile(t[figure_alpha > 0.3], 90)
    else:
        spec_thresh = 0.85

    specular = np.clip((t - spec_thresh) / (1.0 - spec_thresh + 1e-6), 0, 1)
    specular = specular ** 1.5 * 0.5  # soft, moderate highlights
    specular[figure_alpha < 0.2] = 0
    specular = cv2.GaussianBlur(specular, (5, 5), 0)

    # Specular is white
    spec_3ch = np.dstack([specular] * 3)
    figure_surface = figure_surface + spec_3ch * 255.0
    figure_surface = np.clip(figure_surface, 0, 255)

    # ═══════════════════════════════════════════════════════════════
    # STEP 9: Add subtle edge definition
    # ═══════════════════════════════════════════════════════════════
    # We killed all lines, so we need to add back SUBTLE edge definition
    # to separate body parts. Use Canny on the smoothed image (not original!)

    smooth_u8 = smooth.clip(0, 255).astype(np.uint8)
    edges = cv2.Canny(smooth_u8, 30, 80)
    # Only keep edges inside the figure
    edges = cv2.bitwise_and(edges, (figure_alpha * 255).clip(0, 255).astype(np.uint8))
    # Make edges very soft and subtle
    edges_soft = cv2.GaussianBlur(edges.astype(np.float32), (5, 5), 0)
    edges_soft = (edges_soft / edges_soft.max() * 0.15) if edges_soft.max() > 0 else edges_soft

    # Darken along edges slightly (like ambient occlusion on a mannequin)
    edge_3ch = np.dstack([edges_soft] * 3)
    figure_surface = figure_surface * (1.0 - edge_3ch * 0.6)

    # ═══════════════════════════════════════════════════════════════
    # STEP 10: Muscle glow / bloom
    # ═══════════════════════════════════════════════════════════════
    glow = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        glow[:, :, c] = muscle_float * MUSCLE_COLOR_LIGHT[c]

    glow = cv2.GaussianBlur(glow, (25, 25), 0)
    glow = cv2.GaussianBlur(glow, (25, 25), 0)

    # ═══════════════════════════════════════════════════════════════
    # STEP 11: Composite onto background
    # ═══════════════════════════════════════════════════════════════
    bg = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        bg[:, :, c] = BG_COLOR[c]

    alpha_3ch = np.dstack([figure_alpha] * 3)

    canvas = bg * (1.0 - alpha_3ch) + figure_surface * alpha_3ch

    # Add glow
    canvas = canvas + glow * 0.35

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    # ════════════════���══════════════════════════════════════════════
    # STEP 12: Convert RGB order for output
    # ═══════════════════════════════════════════════════════════════
    # figure_surface was built in RGB order, so output directly
    return Image.fromarray(canvas.astype(np.uint8))


def process_gifs():
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".gif")])
    if not files:
        print(f"⚠️  No GIFs found in '{INPUT_DIR}/'")
        return

    total = len(files)
    print(f"🎬 Processing {total} GIFs → Smooth Mannequin...\n")

    for idx, filename in enumerate(files, 1):
        in_path = os.path.join(INPUT_DIR, filename)
        out_path = os.path.join(OUTPUT_DIR, filename)

        try:
            with Image.open(in_path) as im:
                frames = []
                durations = []

                for frame in ImageSequence.Iterator(im):
                    frames.append(process_frame(frame))
                    durations.append(
                        frame.info.get("duration", im.info.get("duration", 100))
                    )

                if frames:
                    frames[0].save(
                        out_path,
                        save_all=True,
                        append_images=frames[1:],
                        loop=0,
                        duration=durations,
                        optimize=False,
                        disposal=2,
                    )
            print(f"  [{idx:>4}/{total}] ✅ {filename}")
        except Exception as e:
            print(f"  [{idx:>4}/{total}] ❌ {filename}: {e}")

    print(f"\n🏁 Done! Results in '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    process_gifs()
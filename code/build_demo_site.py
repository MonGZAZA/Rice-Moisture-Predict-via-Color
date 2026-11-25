"""Generate static assets and JSON data for the interactive GitHub Pages demo website (docs/).

For 12 representative held-out test samples spanning 9.9%–28.8% moisture, exports 6 visual
pipeline stage images (original with crop box, cropped ROI, HSV Hue heatmap, Hue band-stop
denoised, Green segment, Brown segment) into docs/assets/samples/<id>/.

Also exports docs/data/demo_data.json containing the full 38-feature metadata, Pearson r,
training z-score parameters, OLS coefficients, Table I ANOVA, and all 45 held-out test samples.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

from config import (
    CROP_BOX,
    DENOISE_HUE_HIGH,
    DENOISE_HUE_LOW,
    FEATURES_DIR,
    GREEN_HSV_LOWER,
    GREEN_HSV_UPPER,
    OUTPUTS_DIR,
    PICTURES_DIR,
    PROJECT_ROOT,
    SELECTED_FEATURES,
)

DOCS_DIR = PROJECT_ROOT / "docs"
ASSETS_SAMPLES_DIR = DOCS_DIR / "assets" / "samples"
DATA_OUT_DIR = DOCS_DIR / "data"

# 12 featured held-out test sample IDs spanning 9.9% to 28.8% actual moisture
FEATURED_IDS = [245, 290, 8, 22, 27, 44, 69, 281, 95, 117, 143, 175]

FEATURE_GROUPS = {
    "R_skew": ("RGB Statistics", "Skewness of Red channel intensity (Full image)"),
    "V_skew": ("HSV Statistics & Bins", "Skewness of Value/Brightness channel (Full image)"),
    "R_std": ("RGB Statistics", "Standard deviation of Red channel (Green segment)"),
    "Entropy_R": ("Shannon Entropy", "Shannon entropy of Red channel distribution (Green segment)"),
    "G_std": ("RGB Statistics", "Standard deviation of Green channel (Green segment)"),
    "Entropy_G": ("Shannon Entropy", "Shannon entropy of Green channel distribution (Green segment)"),
    "B_std": ("RGB Statistics", "Standard deviation of Blue channel (Green segment)"),
    "Entropy_B": ("Shannon Entropy", "Shannon entropy of Blue channel distribution (Green segment)"),
    "H_std": ("HSV Statistics & Bins", "Standard deviation of Hue angle 0–360° (Green segment)"),
    "H_skew": ("HSV Statistics & Bins", "Skewness of Hue angle distribution (Green segment)"),
    "H_bin0": ("HSV Statistics & Bins", "Hue histogram bin 0: 0°–45° proportion (Green segment)"),
    "H_bin1": ("HSV Statistics & Bins", "Hue histogram bin 1: 45°–90° proportion (Green segment)"),
    "H_bin5": ("HSV Statistics & Bins", "Hue histogram bin 5: 225°–270° proportion (Green segment)"),
    "S_std": ("HSV Statistics & Bins", "Standard deviation of Saturation channel (Green segment)"),
    "S_kurt": ("HSV Statistics & Bins", "Kurtosis of Saturation channel (Green segment)"),
    "S_bin0": ("HSV Statistics & Bins", "Saturation histogram bin 0 (low saturation, Green segment)"),
    "S_bin2": ("HSV Statistics & Bins", "Saturation histogram bin 2 (moderate saturation, Green segment)"),
    "S_bin7": ("HSV Statistics & Bins", "Saturation histogram bin 7 (high saturation, Green segment)"),
    "Entropy_S": ("Shannon Entropy", "Shannon entropy of Saturation channel — top predictor (Green segment)"),
    "V_std": ("HSV Statistics & Bins", "Standard deviation of Value channel (Green segment)"),
    "V_bin0": ("HSV Statistics & Bins", "Value histogram bin 0 (darkest quintile, Green segment)"),
    "V_bin2": ("HSV Statistics & Bins", "Value histogram bin 2 (mid-brightness quintile, Green segment)"),
    "Entropy_V": ("Shannon Entropy", "Shannon entropy of Value channel (Green segment)"),
    "L_std": ("CIELab Color Space", "Standard deviation of L* lightness (Green segment)"),
    "a_mean": ("CIELab Color Space", "Mean of a* green–red opponent channel (Green segment)"),
    "a_std": ("CIELab Color Space", "Standard deviation of a* channel (Green segment)"),
    "b_mean": ("CIELab Color Space", "Mean of b* blue–yellow opponent channel (Green segment)"),
    "b_std": ("CIELab Color Space", "Standard deviation of b* channel (Green segment)"),
    "g_mean": ("Chromaticity & Ratios", "Mean normalized green chromaticity G/(R+G+B) (Green segment)"),
    "R_B_ratio_mean": ("Chromaticity & Ratios", "Mean Red-to-Blue ratio R/B (Green segment)"),
    "G_B_ratio_mean": ("Chromaticity & Ratios", "Mean Green-to-Blue ratio G/B (Green segment)"),
    "ExG_mean": ("Greenness Indices", "Mean Excess Green index 2G − R − B (Green segment)"),
    "ExG_std": ("Greenness Indices", "Standard deviation of Excess Green index (Green segment)"),
    "ExR_std": ("Greenness Indices", "Standard deviation of Excess Red index 1.4R − G (Green segment)"),
    "TGI_mean": ("Greenness Indices", "Mean Triangular Greenness Index −0.5[190(R−G) − 120(R−B)] (Green segment)"),
    "TGI_std": ("Greenness Indices", "Standard deviation of Triangular Greenness Index (Green segment)"),
    "ExB_mean": ("Greenness Indices", "Mean Excess Blue index (Green segment)"),
    "Colorfulness_index": ("Colorfulness", "Hasler & Süsstrunk Colorfulness Index CI (Green segment)"),
}


def _resize_web(img_bgr: np.ndarray, max_width: int = 560) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w <= max_width:
        return img_bgr
    scale = max_width / float(w)
    return cv2.resize(img_bgr, (max_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def _save_jpg(path: Path, img_bgr: np.ndarray, quality: int = 84) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def generate_sample_stages(sample_id: int) -> dict:
    """Generate the 6 visual pipeline stage images and pixel statistics for sample_id."""
    fname = f"{sample_id:04d}.jpg"
    img_path = PICTURES_DIR / fname
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Missing image: {img_path}")

    out_folder = ASSETS_SAMPLES_DIR / f"{sample_id:04d}"
    out_folder.mkdir(parents=True, exist_ok=True)

    # 1. Original image with dashed/highlighted crop box guide
    orig_vis = img_bgr.copy()
    x1, y1, x2, y2 = CROP_BOX
    cv2.rectangle(orig_vis, (x1, y1), (x2, y2), (34, 197, 94), 28)
    _save_jpg(out_folder / "original.jpg", _resize_web(orig_vis))

    # 2. Cropped paddy region
    cropped = img_bgr[y1:y2, x1:x2].copy()
    _save_jpg(out_folder / "cropped.jpg", _resize_web(cropped))

    # 3. HSV Hue channel visualization (matching Fig. 3b)
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    hue_scaled = np.clip((hue.astype(np.float32) / 179.0) * 255.0, 0, 255).astype(np.uint8)
    hue_map = cv2.applyColorMap(hue_scaled, cv2.COLORMAP_HSV)
    _save_jpg(out_folder / "hue_map.jpg", _resize_web(hue_map))

    # 4. Hue band-stop denoising (reject H in [100, 140])
    artifact_mask = (hue >= DENOISE_HUE_LOW) & (hue <= DENOISE_HUE_HIGH)
    denoised = cropped.copy()
    denoised[artifact_mask] = (0, 0, 0)
    _save_jpg(out_folder / "denoised.jpg", _resize_web(denoised))

    valid_mask = ~artifact_mask & ~(
        (denoised[..., 0] == 0) & (denoised[..., 1] == 0) & (denoised[..., 2] == 0)
    )

    # 5 & 6. HSV Color-Splitting: Green vs. Brown complement (matching Fig. 4a & 4b)
    lower = np.array(GREEN_HSV_LOWER, dtype=np.uint8)
    upper = np.array(GREEN_HSV_UPPER, dtype=np.uint8)
    green_raw = cv2.inRange(hsv, lower, upper) > 0
    green_mask = green_raw & valid_mask
    brown_mask = (~green_raw) & valid_mask

    # Render green segment on dark neutral slate and white background option for clarity
    green_img = np.zeros_like(denoised)
    green_img[green_mask] = denoised[green_mask]
    _save_jpg(out_folder / "green.jpg", _resize_web(green_img))

    brown_img = np.zeros_like(denoised)
    brown_img[brown_mask] = denoised[brown_mask]
    _save_jpg(out_folder / "brown.jpg", _resize_web(brown_img))

    total_px = float(cropped.shape[0] * cropped.shape[1])
    valid_px = float(max(1, np.count_nonzero(valid_mask)))
    artifact_pct = float(np.count_nonzero(artifact_mask) / total_px * 100.0)
    green_pct = float(np.count_nonzero(green_mask) / valid_px * 100.0)
    brown_pct = float(np.count_nonzero(brown_mask) / valid_px * 100.0)

    return {
        "artifact_removed_pct": round(artifact_pct, 3),
        "green_segment_pct": round(green_pct, 2),
        "brown_segment_pct": round(brown_pct, 2),
        "images": {
            "original": f"assets/samples/{sample_id:04d}/original.jpg",
            "cropped": f"assets/samples/{sample_id:04d}/cropped.jpg",
            "hue_map": f"assets/samples/{sample_id:04d}/hue_map.jpg",
            "denoised": f"assets/samples/{sample_id:04d}/denoised.jpg",
            "green": f"assets/samples/{sample_id:04d}/green.jpg",
            "brown": f"assets/samples/{sample_id:04d}/brown.jpg",
        },
    }


def main() -> None:
    ASSETS_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_full = pd.read_excel(FEATURES_DIR / "Correlated_FULL.xlsx")
    df_full["name"] = df_full["name"].astype(int)
    df_by_id = df_full.set_index("name")

    bundle = joblib.load(OUTPUTS_DIR / "moisture_ols_model.joblib")
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = list(bundle["feature_cols"])
    beta_orig = list(map(float, bundle["beta_orig"]))
    intercept_orig = float(bundle["intercept_orig"])
    beta_std = list(map(float, model.coef_))
    intercept_std = float(model.intercept_)
    train_mean = list(map(float, scaler.mean_))
    train_std = list(map(float, scaler.scale_))

    pearson_df = pd.read_csv(OUTPUTS_DIR / "pearson_all_243.csv")
    pearson_lookup = {
        (row["segment"], row["feature"]): (float(row["r"]), float(row["p_value"]))
        for _, row in pearson_df.iterrows()
    }
    seg_lookup = {feat: seg for seg, feat in SELECTED_FEATURES}

    features_meta = []
    for idx, col in enumerate(feature_cols):
        seg = seg_lookup.get(col, "Green")
        r_val, p_val = pearson_lookup.get((seg, col), (0.0, 0.0))
        grp, desc = FEATURE_GROUPS.get(col, ("Color Feature", col))
        features_meta.append(
            {
                "index": idx + 1,
                "name": col,
                "segment": seg,
                "group": grp,
                "description": desc,
                "pearson_r": round(r_val, 4),
                "p_value": p_val,
                "train_mean": train_mean[idx],
                "train_std": train_std[idx],
                "beta_std": beta_std[idx],
                "beta_orig": beta_orig[idx],
            }
        )

    # Generate visual pipeline stage images for the 12 featured test samples
    stage_info_by_id = {}
    for idx, sid in enumerate(FEATURED_IDS, 1):
        print(f"[{idx:02d}/{len(FEATURED_IDS)}] Rendering pipeline stages for {sid:04d}.jpg...")
        stage_info_by_id[sid] = generate_sample_stages(sid)

    # Build records for all 45 held-out test samples (sorted by actual moisture)
    test_preds = pd.read_csv(OUTPUTS_DIR / "test_predictions.csv")
    test_samples = []
    for rank_idx, row in test_preds.iterrows():
        sid = int(row["name"])
        actual = float(row["actual"])
        predicted = float(row["predicted"])
        abs_err = float(row["abs_error"])
        pct_err = float(row["pct_error"])

        feat_row = df_by_id.loc[sid, feature_cols]
        raw_vals = [float(feat_row[c]) for c in feature_cols]
        z_vals = [(raw_vals[i] - train_mean[i]) / train_std[i] for i in range(len(feature_cols))]
        contribs = [z_vals[i] * beta_std[i] for i in range(len(feature_cols))]

        rec = {
            "test_rank": int(rank_idx),
            "id": sid,
            "filename": f"{sid:04d}.jpg",
            "actual": round(actual, 2),
            "predicted": round(predicted, 3),
            "abs_error": round(abs_err, 3),
            "pct_error": round(pct_err, 2),
            "featured": sid in stage_info_by_id,
            "raw_features": [round(v, 6) for v in raw_vals],
            "z_scores": [round(z, 4) for z in z_vals],
            "z_contributions": [round(c, 4) for c in contribs],
        }
        if sid in stage_info_by_id:
            rec.update(stage_info_by_id[sid])
        test_samples.append(rec)

    anova_df = pd.read_csv(OUTPUTS_DIR / "regression_anova_table.csv")
    anova_rows = anova_df.where(pd.notna(anova_df), None).to_dict(orient="records")

    payload = {
        "paper": {
            "title": "Developing Prediction Model for Estimating Moisture of Rice Paddy using Correlation Between Color Values of Rice Paddy and Moisture Content",
            "authors": ["Kirati Pothidang", "Ritchanon Bunseng", "Waiyawat Suansawan"],
            "affiliation": "Department of Mathematics, Princess Chulabhorn Science High School Phetchaburi, Thailand",
            "venue": "IEEE iSAI-NLP 2025",
            "ieee_url": "https://ieeexplore.ieee.org/document/11320486",
            "github_url": "https://github.com/MonGZAZA/Rice-Moisture-Predict-via-Color",
            "camera": "FUJIFILM X100T (4896×3264 px, ISO 200, f/11, 20×20 cm lightbox)",
            "crop_box": list(CROP_BOX),
            "denoise_hue_band": [DENOISE_HUE_LOW, DENOISE_HUE_HIGH],
            "green_hsv_range": [list(GREEN_HSV_LOWER), list(GREEN_HSV_UPPER)],
        },
        "model": {
            "intercept_std": intercept_std,
            "intercept_orig": intercept_orig,
            "metrics": bundle["metrics"],
            "anova": anova_rows,
            "features": features_meta,
        },
        "featured_ids": FEATURED_IDS,
        "test_samples": test_samples,
    }

    out_json = DATA_OUT_DIR / "demo_data.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {out_json} ({out_json.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

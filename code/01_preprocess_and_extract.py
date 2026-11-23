"""Step 1 — Image Preprocessing, Color-Splitting & Color Feature Extraction.

Implements Sections IV.B, IV.C, and IV.D of the IEEE iSAI-NLP 2025 paper:
  1. Crop borders to retain only the paddy region.
  2. Hue-only band-stop denoising (reject H in [100, 140] in OpenCV HSV).
  3. HSV color-splitting into Green (H in [25,85], S in [35,255], V in [35,255])
     and Brown (complement of Green) segments alongside the Full image.
  4. Extract 81 color features per segment (243 total features per sample):
       - RGB mean, std, skew, kurtosis, Shannon entropy (15)
       - HSV mean, std, skew, kurtosis, entropy + 8 H-bins, 8 S-bins, 5 V-bins (36)
       - CIELab L*, a*, b* mean, std, skew, kurtosis (12)
       - Chromaticity r, g (mean, std) + R/G, R/B, G/B mean ratios (7)
       - Greenness indices: ExG, ExR, VARI, TGI, ExB (mean, std) (10)
       - Hasler & Süsstrunk Colorfulness Index CI (1)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy
from scipy.stats import kurtosis, skew

from config import (
    CROP_BOX,
    DENOISE_HUE_HIGH,
    DENOISE_HUE_LOW,
    GREEN_HSV_LOWER,
    GREEN_HSV_UPPER,
    MAPPING_ALL_512,
    OUTPUTS_DIR,
    PICTURES_DIR,
)


def crop_paddy_region(img_bgr: np.ndarray, crop_box: Tuple[int, int, int, int] = CROP_BOX) -> np.ndarray:
    """Crop the lightbox borders if the image matches or exceeds the crop box."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = crop_box
    if w >= x2 and h >= y2:
        return img_bgr[y1:y2, x1:x2].copy()
    return img_bgr.copy()


def hue_bandstop_denoise(
    img_bgr: np.ndarray,
    low_h: int = DENOISE_HUE_LOW,
    high_h: int = DENOISE_HUE_HIGH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove non-paddy artifacts with Hue in [low_h, high_h] (OpenCV scale 0-179).

    Returns (denoised_bgr, valid_pixel_mask).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    artifact_mask = (hue >= low_h) & (hue <= high_h)
    denoised = img_bgr.copy()
    denoised[artifact_mask] = (0, 0, 0)
    valid_mask = ~artifact_mask & ~((denoised[..., 0] == 0) & (denoised[..., 1] == 0) & (denoised[..., 2] == 0))
    return denoised, valid_mask


def split_green_brown(
    denoised_bgr: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split denoised paddy image into Green and Brown boolean pixel masks (Sec. IV.C)."""
    hsv = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(GREEN_HSV_LOWER, dtype=np.uint8)
    upper = np.array(GREEN_HSV_UPPER, dtype=np.uint8)
    green_mask = (cv2.inRange(hsv, lower, upper) > 0) & valid_mask
    brown_mask = (~green_mask) & valid_mask
    return green_mask, brown_mask


def _basic_stats(vals: np.ndarray) -> Tuple[float, float, float, float]:
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    sk = float(skew(vals)) if vals.size > 2 else 0.0
    ku = float(kurtosis(vals, fisher=False)) if vals.size > 3 else 0.0
    return mean, std, sk, ku


def _hist_bins(vals: np.ndarray, bins: int, range_max: float) -> list[float]:
    h, _ = np.histogram(vals, bins=bins, range=(0.0, range_max))
    total = h.sum()
    if total == 0:
        return [0.0] * bins
    return (h.astype(np.float64) / total).tolist()


def _shannon_entropy(vals: np.ndarray, bins: int = 256, range_max: float = 256.0) -> float:
    h, _ = np.histogram(vals, bins=bins, range=(0.0, range_max))
    total = h.sum()
    if total == 0:
        return 0.0
    p = h / total
    p = p[p > 0]
    return float(sp_entropy(p, base=2))


def extract_color_features(
    img_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    hsv: Optional[np.ndarray] = None,
    lab: Optional[np.ndarray] = None,
) -> Optional[Dict[str, float]]:
    """Extract the 81 color features (Sec. IV.D) over masked non-black pixels."""
    b, g, r = cv2.split(img_bgr)
    if mask is None:
        mask = ~((r == 0) & (g == 0) & (b == 0))
    if not np.any(mask):
        return None

    R = r[mask].astype(np.float64)
    G = g[mask].astype(np.float64)
    B = b[mask].astype(np.float64)

    feats: Dict[str, float] = {}

    # 1) RGB stats + Shannon entropy (15 features)
    for cname, vals in (("R", R), ("G", G), ("B", B)):
        mean, std, sk, ku = _basic_stats(vals)
        feats[f"{cname}_mean"] = mean
        feats[f"{cname}_std"] = std
        feats[f"{cname}_skew"] = sk
        feats[f"{cname}_kurt"] = ku
        feats[f"Entropy_{cname}"] = _shannon_entropy(vals, bins=256, range_max=256.0)

    # 2) HSV stats, normalized histogram bins (8/8/5), and entropy (36 features)
    if hsv is None:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H = hsv[..., 0][mask].astype(np.float64) * 2.0  # degrees [0, 358]
    S = hsv[..., 1][mask].astype(np.float64)
    V = hsv[..., 2][mask].astype(np.float64)
    for name, vals, rng, nbins, ent_rng in (
        ("H", H, 360.0, 8, 360.0),
        ("S", S, 255.0, 8, 256.0),
        ("V", V, 255.0, 5, 256.0),
    ):
        mean, std, sk, ku = _basic_stats(vals)
        feats[f"{name}_mean"] = mean
        feats[f"{name}_std"] = std
        feats[f"{name}_skew"] = sk
        feats[f"{name}_kurt"] = ku
        for idx, bin_val in enumerate(_hist_bins(vals, nbins, rng)):
            feats[f"{name}_bin{idx}"] = bin_val
        feats[f"Entropy_{name}"] = _shannon_entropy(vals, bins=256, range_max=ent_rng)

    # 3) CIELab stats (12 features, OpenCV 8-bit Lab convention matching paper pipeline)
    if lab is None:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L_ch = lab[..., 0][mask].astype(np.float64)
    a_ch = lab[..., 1][mask].astype(np.float64)
    b_ch = lab[..., 2][mask].astype(np.float64)
    for cname, vals in (("L", L_ch), ("a", a_ch), ("b", b_ch)):
        mean, std, sk, ku = _basic_stats(vals)
        feats[f"{cname}_mean"] = mean
        feats[f"{cname}_std"] = std
        feats[f"{cname}_skew"] = sk
        feats[f"{cname}_kurt"] = ku

    # 4 & 5) Chromaticity (r, g) & RGB channel ratios (7 features)
    sum_rgb = R + G + B
    r_frac = np.where(sum_rgb > 0, R / np.where(sum_rgb > 0, sum_rgb, 1.0), 0.0)
    g_frac = np.where(sum_rgb > 0, G / np.where(sum_rgb > 0, sum_rgb, 1.0), 0.0)
    feats["r_mean"], feats["r_std"], _, _ = _basic_stats(r_frac)
    feats["g_mean"], feats["g_std"], _, _ = _basic_stats(g_frac)
    feats["R_G_ratio_mean"] = float(np.mean(R / np.where(G != 0, G, 1.0)))
    feats["R_B_ratio_mean"] = float(np.mean(R / np.where(B != 0, B, 1.0)))
    feats["G_B_ratio_mean"] = float(np.mean(G / np.where(B != 0, B, 1.0)))

    # 6–8) Greenness / Vegetation indices: ExG, ExR, VARI, TGI, ExB (10 features)
    exg = 2.0 * G - R - B
    exr = 1.4 * R - G
    vari_denom = G + R - B
    vari = (G - R) / np.where(vari_denom != 0, vari_denom, 1.0)
    tgi = -0.5 * (190.0 * (R - G) - 120.0 * (R - B))
    exb = 2.0 * B - R - G
    for name, arr in (("ExG", exg), ("ExR", exr), ("VARI", vari), ("TGI", tgi), ("ExB", exb)):
        mean, std, _, _ = _basic_stats(arr)
        feats[f"{name}_mean"] = mean
        feats[f"{name}_std"] = std

    # 9) Hasler & Süsstrunk Colorfulness Index (1 feature)
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_rg = float(np.std(rg, ddof=1)) if rg.size > 1 else 0.0
    std_yb = float(np.std(yb, ddof=1)) if yb.size > 1 else 0.0
    mean_rg = float(np.mean(rg))
    mean_yb = float(np.mean(yb))
    feats["Colorfulness_index"] = float(
        np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
    )

    return feats


def _crop_to_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]


def _jpg_roundtrip(img_bgr: np.ndarray, quality: int = 95) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode JPEG buffer")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def extract_all_segments_from_image(
    img_path: Path,
    paper_compat: bool = False,
) -> Dict[str, Optional[Dict[str, float]]]:
    """Run full preprocessing + green/brown split + feature extraction for a single image.

    Parameters
    ----------
    img_path : Path
        Path to the RGB paddy image.
    paper_compat : bool, default False
        If True, mirrors the original RiTcHy_Code.ipynb Colab pipeline (where
        intermediate cropped, denoised, and green/brown masked images were saved
        as JPEG files before feature extraction). Required when running inference
        with the exact paper_296 model trained on the published CSVs.
        If False (recommended for training new models), keeps masks lossless in memory.
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    cropped = crop_paddy_region(img_bgr)

    if paper_compat:
        # Reproduce RiTcHy_Code.ipynb Cell 2 (PIL jpg q=75) -> Cell 10 (cv2 jpg q=95) -> Cell 13 (cv2 jpg q=95)
        cropped_j = _jpg_roundtrip(cropped, quality=75)
        hsv_c = cv2.cvtColor(cropped_j, cv2.COLOR_BGR2HSV)
        denoised = cropped_j.copy()
        denoised[(hsv_c[..., 0] >= DENOISE_HUE_LOW) & (hsv_c[..., 0] <= DENOISE_HUE_HIGH)] = (0, 0, 0)
        denoised = _jpg_roundtrip(denoised, quality=95)

        hsv_d = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
        lower = np.array(GREEN_HSV_LOWER, dtype=np.uint8)
        upper = np.array(GREEN_HSV_UPPER, dtype=np.uint8)
        gmask = cv2.inRange(hsv_d, lower, upper)
        bmask = cv2.bitwise_not(gmask)

        green_crop = _jpg_roundtrip(_crop_to_mask(cv2.bitwise_and(denoised, denoised, mask=gmask), gmask), quality=95)
        brown_crop = _jpg_roundtrip(_crop_to_mask(cv2.bitwise_and(denoised, denoised, mask=bmask), bmask), quality=95)

        return {
            "All": extract_color_features(denoised, mask=None),
            "Green": extract_color_features(green_crop, mask=None),
            "Brown": extract_color_features(brown_crop, mask=None),
        }

    denoised, valid_mask = hue_bandstop_denoise(cropped)
    green_mask, brown_mask = split_green_brown(denoised, valid_mask)

    # Pre-compute HSV and Lab once per image for speed
    hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2Lab)

    return {
        "All": extract_color_features(denoised, valid_mask, hsv=hsv, lab=lab),
        "Green": extract_color_features(denoised, green_mask, hsv=hsv, lab=lab),
        "Brown": extract_color_features(denoised, brown_mask, hsv=hsv, lab=lab),
    }


def run_extraction(
    mapping_csv: Path,
    pictures_dir: Path,
    out_dir: Path,
    limit: Optional[int] = None,
) -> Dict[str, Path]:
    """Extract All/Green/Brown color CSVs for every sample listed in mapping_csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_df = pd.read_csv(mapping_csv)
    if limit is not None:
        mapping_df = mapping_df.head(limit)

    records = {"All": [], "Green": [], "Brown": []}
    total = len(mapping_df)

    for idx, row in mapping_df.iterrows():
        filename = str(row["name"])
        moisture = float(row["moisture%"])
        sample_id = Path(filename).stem
        img_path = pictures_dir / filename

        seg_feats = extract_all_segments_from_image(img_path)
        for seg_name, feats in seg_feats.items():
            if feats is None:
                continue
            rec = {"name": int(sample_id) if sample_id.isdigit() else sample_id, "moisture%": moisture}
            rec.update(feats)
            records[seg_name].append(rec)

        if (idx + 1) % 25 == 0 or (idx + 1) == total:
            print(f"[Extract] Processed {idx + 1}/{total} images ({filename})")

    saved_paths: Dict[str, Path] = {}
    for seg_name, rows in records.items():
        df_seg = pd.DataFrame(rows).sort_values("name").reset_index(drop=True)
        out_path = out_dir / f"{seg_name}_color.csv"
        df_seg.to_csv(out_path, index=False)
        saved_paths[seg_name] = out_path
        print(f"Saved {out_path} ({df_seg.shape[0]} rows x {df_seg.shape[1]} cols)")

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract rice paddy color features (paper Sec. IV.B–D).")
    parser.add_argument("--mapping", type=Path, default=MAPPING_ALL_512, help="Path to mapping.csv")
    parser.add_argument("--pictures", type=Path, default=PICTURES_DIR, help="Directory of paddy images")
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR / "extracted", help="Output folder for CSVs")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of images to process")
    args = parser.parse_args()

    run_extraction(args.mapping, args.pictures, args.out_dir, limit=args.limit)


if __name__ == "__main__":
    main()

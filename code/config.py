"""Configuration constants for the Rice Paddy Moisture Prediction pipeline."""

from pathlib import Path

# Project directories
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PICTURES_DIR = DATA_DIR / "pictures"
FEATURES_DIR = DATA_DIR / "features"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ground-truth moisture labels
MAPPING_CSV = DATA_DIR / "mapping.csv"

# Cropping box (x1, y1, x2, y2) for 4896x3264 lightbox images
CROP_BOX = (586, 0, 4252, 3164)

# Hue-only band-stop reject range in OpenCV HSV scale [0, 179]
DENOISE_HUE_LOW = 100
DENOISE_HUE_HIGH = 140

# HSV thresholding for green segment (OpenCV H:[0,179], S:[0,255], V:[0,255])
GREEN_HSV_LOWER = (25, 35, 35)
GREEN_HSV_UPPER = (85, 255, 255)

# Pearson correlation feature screening thresholds
CORR_THRESHOLD = 0.80
PVALUE_THRESHOLD = 0.01

# Stratified train/test split parameters
N_STRATIFY_DECILES = 10
TEST_SIZE = 0.15
RANDOM_STATE = 42

# 38 retained features (|r| >= 0.80, p < 0.01):
# 2 from full ("All") segment + 36 from "Green" segment
SELECTED_FEATURES = [
    ("All", "R_skew"),
    ("All", "V_skew"),
    ("Green", "R_std"),
    ("Green", "Entropy_R"),
    ("Green", "G_std"),
    ("Green", "Entropy_G"),
    ("Green", "B_std"),
    ("Green", "Entropy_B"),
    ("Green", "H_std"),
    ("Green", "H_skew"),
    ("Green", "H_bin0"),
    ("Green", "H_bin1"),
    ("Green", "H_bin5"),
    ("Green", "S_std"),
    ("Green", "S_kurt"),
    ("Green", "S_bin0"),
    ("Green", "S_bin2"),
    ("Green", "S_bin7"),
    ("Green", "Entropy_S"),
    ("Green", "V_std"),
    ("Green", "V_bin0"),
    ("Green", "V_bin2"),
    ("Green", "Entropy_V"),
    ("Green", "L_std"),
    ("Green", "a_mean"),
    ("Green", "a_std"),
    ("Green", "b_mean"),
    ("Green", "b_std"),
    ("Green", "g_mean"),
    ("Green", "R_B_ratio_mean"),
    ("Green", "G_B_ratio_mean"),
    ("Green", "ExG_mean"),
    ("Green", "ExG_std"),
    ("Green", "ExR_std"),
    ("Green", "TGI_mean"),
    ("Green", "TGI_std"),
    ("Green", "ExB_mean"),
    ("Green", "Colorfulness_index"),
]

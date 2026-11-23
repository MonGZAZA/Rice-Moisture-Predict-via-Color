"""Master Pipeline Runner for Rice Paddy Moisture Prediction.

Modes:
  - train (default): Runs Pearson correlation screening (38 retained features)
      and decile-stratified 85/15 OLS training/evaluation using data/features/,
      exporting ANOVA, metrics, equation, and actual_vs_predicted.png to outputs/.
  - extract_and_train: Runs full raw-image preprocessing & feature extraction
      from data/pictures/ using data/mapping.csv before correlation screening
      and OLS training/evaluation.
"""

from __future__ import annotations

import argparse
from importlib import import_module

from config import (
    FEATURES_DIR,
    MAPPING_CSV,
    OUTPUTS_DIR,
    PICTURES_DIR,
    RANDOM_STATE,
)

step1 = import_module("01_preprocess_and_extract")
step2 = import_module("02_correlation_select")
step3 = import_module("03_train_and_evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Rice Paddy Moisture Prediction pipeline.")
    parser.add_argument(
        "--mode",
        choices=("train", "extract_and_train"),
        default="train",
        help="'train' runs correlation screening and OLS training/evaluation on data/features/; "
        "'extract_and_train' runs raw image extraction from data/pictures/ first.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.mode == "train":
        out_dir = OUTPUTS_DIR
        print("[Pipeline] Mode: train & evaluate")
        step2.run_correlation_selection(FEATURES_DIR, out_dir)
        xlsx_data = FEATURES_DIR / "Correlated_FULL.xlsx"
        step3.train_and_evaluate(
            data_path=xlsx_data if xlsx_data.exists() else (out_dir / "Correlated_FULL.csv"),
            out_dir=out_dir,
            manifest_path=out_dir / "selected_feature_manifest.csv",
            random_state=args.seed,
            extraction_mode="jpeg_compat",
        )
    else:
        out_dir = OUTPUTS_DIR
        extract_dir = FEATURES_DIR
        print(f"[Pipeline] Mode: full extract_and_train ({MAPPING_CSV.name})")
        step1.run_extraction(MAPPING_CSV, PICTURES_DIR, extract_dir)
        step2.run_correlation_selection(extract_dir, out_dir)
        step3.train_and_evaluate(
            data_path=out_dir / "Correlated_FULL.csv",
            out_dir=out_dir,
            manifest_path=out_dir / "selected_feature_manifest.csv",
            random_state=args.seed,
            extraction_mode="clean",
        )


if __name__ == "__main__":
    main()

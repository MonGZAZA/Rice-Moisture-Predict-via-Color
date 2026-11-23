"""Master Pipeline Runner for Rice Paddy Moisture Prediction.

Modes:
  - train (default): Runs Pearson correlation screening (38 retained features)
      and decile-stratified 85/15 OLS training/evaluation on the 296-sample
      extracted feature set, exporting ANOVA, metrics, equation, and Fig. 6.
      (Also accepts 'paper' as an alias.)
  - extract_and_train: Runs full raw-image preprocessing & feature extraction
      from data/pictures/ (either 296 or 512 samples) before correlation
      screening and OLS training/evaluation.
"""

from __future__ import annotations

import argparse
from importlib import import_module

from config import (
    EXTRACTED_PAPER_DIR,
    MAPPING_ALL_512,
    MAPPING_PAPER_296,
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
        choices=("train", "paper", "extract_and_train"),
        default="train",
        help="'train' (alias 'paper') runs correlation screening and OLS training/evaluation on extracted 296-sample features; "
        "'extract_and_train' runs raw image extraction from data/pictures/ first.",
    )
    parser.add_argument(
        "--dataset",
        choices=("296", "512"),
        default="296",
        help="Dataset size for extract_and_train mode (296 = core study set, 512 = extended image set).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.mode in ("train", "paper"):
        out_dir = OUTPUTS_DIR / "paper_296"
        print("[Pipeline] Mode: train & evaluate (N=296)")
        step2.run_correlation_selection(EXTRACTED_PAPER_DIR, out_dir)
        xlsx_data = EXTRACTED_PAPER_DIR / "Correlated_FULL.xlsx"
        step3.train_and_evaluate(
            data_path=xlsx_data if xlsx_data.exists() else (out_dir / "Correlated_FULL.csv"),
            out_dir=out_dir,
            manifest_path=out_dir / "selected_feature_manifest.csv",
            random_state=args.seed,
            extraction_mode="paper_compat",
        )
    else:
        mapping_csv = MAPPING_PAPER_296 if args.dataset == "296" else MAPPING_ALL_512
        out_dir = OUTPUTS_DIR / f"fresh_extract_{args.dataset}"
        extract_dir = out_dir / "extracted_features"
        print(f"[Pipeline] Mode: full extract_and_train (N={args.dataset}, mapping={mapping_csv.name})")
        step1.run_extraction(mapping_csv, PICTURES_DIR, extract_dir)
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

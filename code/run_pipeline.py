"""Master Pipeline Runner for the IEEE iSAI-NLP 2025 Rice Paddy Moisture Model.

Modes:
  - paper (default): Reproduces the IEEE paper's exact 296-sample results
      (correlation screening -> 38 features -> decile-stratified 85/15 OLS ->
      RMSE=2.270, MAE=1.703, MAPE=9.75%, R^2=0.840, Table I ANOVA, Fig. 6).
  - extract_and_train: Runs full raw-image preprocessing & feature extraction
      from data/pictures/ (either 296 paper samples or all 512 samples) before
      correlation screening and OLS training/evaluation.
"""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

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
        choices=("paper", "extract_and_train"),
        default="paper",
        help="'paper' reproduces exact IEEE paper tables/plots from extracted 296-sample features; "
        "'extract_and_train' runs raw image extraction from data/pictures/ first.",
    )
    parser.add_argument(
        "--dataset",
        choices=("296", "512"),
        default="296",
        help="Dataset size for extract_and_train mode (296 = paper subset, 512 = full updated dataset).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.mode == "paper":
        out_dir = OUTPUTS_DIR / "paper_296"
        print("[Pipeline] Mode: paper reproduction (N=296)")
        step2.run_correlation_selection(EXTRACTED_PAPER_DIR, out_dir)
        # Use the exact Correlated_FULL.xlsx workbook from the paper to match
        # IEEE floating-point precision (reproduces RMSE=2.270, MAE=1.703, MAPE=9.75%, R^2=0.840)
        paper_data = EXTRACTED_PAPER_DIR / "Correlated_FULL.xlsx"
        step3.train_and_evaluate(
            data_path=paper_data if paper_data.exists() else (out_dir / "Correlated_FULL.csv"),
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

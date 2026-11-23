"""End-to-End Moisture Prediction Inference Script.

Predicts wet-basis rice paddy moisture content (%) for one or more RGB images
using the trained standardized OLS model bundle exported by 03_train_and_evaluate.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

from config import OUTPUTS_DIR
from importlib import import_module

extract_mod = import_module("01_preprocess_and_extract")


def predict_image(img_path: Path, bundle: dict) -> float:
    """Extract All/Green/Brown color features for img_path and predict moisture%."""
    paper_compat = bundle.get("extraction_mode", "paper_compat") == "paper_compat"
    seg_feats = extract_mod.extract_all_segments_from_image(img_path, paper_compat=paper_compat)
    feature_specs = bundle["feature_specs"]

    vec: List[float] = []
    for col_name, seg_name, raw_feat in feature_specs:
        feats = seg_feats.get(seg_name)
        if feats is None or raw_feat not in feats:
            raise ValueError(
                f"Could not extract feature '{raw_feat}' from '{seg_name}' segment for {img_path.name} "
                f"(segment may have zero valid pixels)."
            )
        vec.append(float(feats[raw_feat]))

    x_df = pd.DataFrame([vec], columns=bundle["feature_cols"])
    x_std = bundle["scaler"].transform(x_df)
    pred = float(bundle["model"].predict(x_std)[0])
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict rice paddy moisture% from RGB image(s).")
    parser.add_argument("images", nargs="+", type=Path, help="Path(s) to paddy image file(s)")
    parser.add_argument(
        "--model",
        type=Path,
        default=OUTPUTS_DIR / "paper_296" / "moisture_ols_model.joblib",
        help="Path to trained moisture_ols_model.joblib",
    )
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    for img_path in args.images:
        pred = predict_image(img_path, bundle)
        print(f"{img_path.name:20s} -> Predicted Moisture: {pred:.2f}%")


if __name__ == "__main__":
    main()

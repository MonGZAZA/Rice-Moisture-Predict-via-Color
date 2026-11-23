"""Step 2 — Pearson Correlation Analysis & Feature Screening.

Computes Pearson correlation coefficient (r) and two-sided p-value against
moisture% for all features across All, Green, and Brown segments (243 total),
retaining features with |r| >= 0.80 and p < 0.01 into Correlated_FULL.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from scipy.stats import pearsonr

from config import (
    CORR_THRESHOLD,
    FEATURES_DIR,
    OUTPUTS_DIR,
    PVALUE_THRESHOLD,
    SELECTED_FEATURES,
)


def run_correlation_selection(
    features_dir: Path,
    out_dir: Path,
    corr_threshold: float = CORR_THRESHOLD,
    pvalue_threshold: float = PVALUE_THRESHOLD,
    preserve_feature_order: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson correlations across All/Green/Brown CSVs and filter features."""
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = {}
    for seg in ("All", "Green", "Brown"):
        csv_path = features_dir / f"{seg}_color.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing feature CSV: {csv_path}")
        df_seg = pd.read_csv(csv_path)
        df_seg = df_seg.sort_values("name").reset_index(drop=True)
        segments[seg] = df_seg

    # Compute Pearson r and p-value across all 243 features (81 x 3)
    corr_rows = []
    for seg, df_seg in segments.items():
        y = df_seg["moisture%"].astype(float)
        feat_cols = [c for c in df_seg.columns if c not in ("name", "moisture%")]
        for col in feat_cols:
            x = df_seg[col].astype(float)
            if x.nunique() <= 1:
                r_val, p_val = 0.0, 1.0
            else:
                r_val, p_val = pearsonr(x, y)
            selected = (abs(r_val) >= corr_threshold) and (p_val < pvalue_threshold)
            corr_rows.append(
                {
                    "segment": seg,
                    "feature": col,
                    "r": float(r_val),
                    "abs_r": abs(float(r_val)),
                    "p_value": float(p_val),
                    "selected": bool(selected),
                }
            )

    corr_df = pd.DataFrame(corr_rows).sort_values("abs_r", ascending=False).reset_index(drop=True)
    corr_df.to_csv(out_dir / "pearson_all_243.csv", index=False)

    sel_df = corr_df[corr_df["selected"]].copy().reset_index(drop=True)
    sel_df.to_csv(out_dir / "selected_features_summary.csv", index=False)

    selected_pairs: List[Tuple[str, str]] = list(zip(sel_df["segment"], sel_df["feature"]))

    # Preserve canonical column order when matching the 38-feature subset
    if preserve_feature_order and set(selected_pairs) == set(SELECTED_FEATURES):
        ordered_pairs = list(SELECTED_FEATURES)
    else:
        ordered_pairs = selected_pairs

    # Build combined feature table indexed by sample name & moisture%
    base = segments["All"][["name", "moisture%"]].copy()
    used_col_names = set()
    feature_manifest = []
    for seg, feat in ordered_pairs:
        col_name = feat if feat not in used_col_names else f"{seg}_{feat}"
        used_col_names.add(col_name)
        seg_lookup = segments[seg].set_index("name")[feat]
        base[col_name] = base["name"].map(seg_lookup)
        feature_manifest.append({"column_name": col_name, "segment": seg, "feature": feat})

    base = base.dropna().reset_index(drop=True)
    base.to_csv(out_dir / "Correlated_FULL.csv", index=False)
    pd.DataFrame(feature_manifest).to_csv(out_dir / "selected_feature_manifest.csv", index=False)

    seg_counts = sel_df["segment"].value_counts().to_dict()
    print(f"[Correlation] Studied variables: {len(corr_df)}")
    print(
        f"[Correlation] Retained variables (|r| >= {corr_threshold}, p < {pvalue_threshold}): "
        f"{len(sel_df)} -> {seg_counts}"
    )
    print(f"[Correlation] Top 5 predictors:")
    for _, r in sel_df.head(5).iterrows():
        print(f"    {r['segment']:5s} | {r['feature']:20s} | r = {r['r']:+.4f} | p = {r['p_value']:.2e}")

    return corr_df, base


def main() -> None:
    parser = argparse.ArgumentParser(description="Pearson correlation feature screening.")
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=FEATURES_DIR,
        help="Directory containing All_color.csv, Green_color.csv, Brown_color.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR, help="Output directory")
    parser.add_argument("--corr-thresh", type=float, default=CORR_THRESHOLD)
    parser.add_argument("--p-thresh", type=float, default=PVALUE_THRESHOLD)
    args = parser.parse_args()

    run_correlation_selection(args.features_dir, args.out_dir, args.corr_thresh, args.p_thresh)


if __name__ == "__main__":
    main()

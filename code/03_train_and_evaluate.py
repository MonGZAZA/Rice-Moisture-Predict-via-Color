"""Step 3 — Stratified Split, Standardized OLS Training, Back-Transformation & Evaluation.

Implements Sections IV.F, IV.G, V.B, V.C, Table I, and Fig. 6 of the
IEEE iSAI-NLP 2025 paper:
  - 10-decile stratified split: 85% development/training set (N=251),
    15% held-out testing set (N=45) on the 296-sample paper dataset.
  - Z-score standardization fitted strictly on the training set.
  - Ordinary Least Squares (OLS) regression with intercept term.
  - Back-transformation of standardized coefficients into original feature units:
        y_hat = beta_0 + sum(beta_j * x_j)
  - Evaluation on held-out test set: RMSE, MAE, MAPE (%), R^2.
  - Regression ANOVA table (Table I): SS, df, MS, F-statistic, p-value.
  - Actual vs. Predicted Moisture scatter plot (Fig. 6).
  - Exports trained model bundle (joblib), operational equation, ANOVA CSV,
    test predictions CSV, and metrics JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    EXTRACTED_PAPER_DIR,
    N_STRATIFY_DECILES,
    OUTPUTS_DIR,
    PAPER_SELECTED_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)


def load_dataset(data_path: Path) -> pd.DataFrame:
    if data_path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(data_path)
    return pd.read_csv(data_path)


def train_and_evaluate(
    data_path: Path,
    out_dir: Path,
    manifest_path: Path | None = None,
    n_deciles: int = N_STRATIFY_DECILES,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    extraction_mode: str = "paper_compat",
) -> Dict[str, Any]:
    """Run the IEEE paper's training, ANOVA, back-transformation, and held-out evaluation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(data_path)

    feature_cols = [c for c in df.columns if c not in ("name", "moisture%")]
    X = df[feature_cols].astype(float)
    y = df["moisture%"].astype(float).values
    names = df["name"].values if "name" in df.columns else np.arange(1, len(df) + 1)

    # Sec. IV.F: 10 equal-frequency deciles for continuous-target stratified split
    deciles = pd.qcut(y, q=n_deciles, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test, name_train, name_test = train_test_split(
        X,
        y,
        names,
        test_size=test_size,
        stratify=deciles,
        random_state=random_state,
    )

    # Standardize predictors using training-set mean & std (z-scores)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Fit OLS on standardized development set (with intercept)
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train_std, y_train)

    # Condition number of standardized design matrix (with intercept column)
    design_std = np.column_stack([np.ones(len(X_train_std)), X_train_std])
    cond_number = float(np.linalg.cond(design_std))

    # Back-transform standardized OLS coefficients to original feature units (Eq. 12)
    beta_orig = model.coef_ / scaler.scale_
    intercept_orig = float(model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_))

    # Predict on held-out test set (both standardized and back-transformed)
    y_test_pred = model.predict(X_test_std)
    y_test_pred_orig = X_test.values @ beta_orig + intercept_orig
    max_bt_diff = float(np.max(np.abs(y_test_pred - y_test_pred_orig)))

    # Sec. IV.G / V.C: Held-out test metrics
    rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    mae = float(mean_absolute_error(y_test, y_test_pred))
    mape = float(np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100.0)
    r2 = float(r2_score(y_test, y_test_pred))

    # Table I: Regression ANOVA on development set (N_train, p predictors)
    y_train_pred = model.predict(X_train_std)
    n_train = len(y_train)
    p = len(feature_cols)
    ss_reg = float(np.sum((y_train_pred - np.mean(y_train)) ** 2))
    ss_res = float(np.sum((y_train - y_train_pred) ** 2))
    ss_tot = float(np.sum((y_train - np.mean(y_train)) ** 2))
    df_reg = p
    df_res = n_train - p - 1
    df_tot = n_train - 1
    ms_reg = ss_reg / df_reg
    ms_res = ss_res / df_res
    f_stat = ms_reg / ms_res
    p_value = float(sp_stats.f.sf(f_stat, df_reg, df_res))
    r2_train = float(r2_score(y_train, y_train_pred))
    adj_r2_train = 1.0 - (1.0 - r2_train) * (n_train - 1) / df_res

    anova_df = pd.DataFrame(
        [
            {"Source": "Regression", "SS": round(ss_reg, 3), "df": df_reg, "MS": round(ms_reg, 4), "F": round(f_stat, 3), "p-value": p_value},
            {"Source": "Residual", "SS": round(ss_res, 3), "df": df_res, "MS": round(ms_res, 4), "F": np.nan, "p-value": np.nan},
            {"Source": "Total", "SS": round(ss_tot, 3), "df": df_tot, "MS": np.nan, "F": np.nan, "p-value": np.nan},
        ]
    )
    anova_df.to_csv(out_dir / "regression_anova_table.csv", index=False)

    # Save coefficients & operational equation (Eq. 12)
    coef_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "beta_standardized": model.coef_,
            "beta_original_units": beta_orig,
            "train_mean": scaler.mean_,
            "train_std": scaler.scale_,
        }
    )
    coef_df.to_csv(out_dir / "model_coefficients.csv", index=False)

    eq_terms = [f"{intercept_orig:.6f}"]
    for feat, coef in zip(feature_cols, beta_orig):
        sign = "+" if coef >= 0 else "-"
        eq_terms.append(f"{sign} {abs(coef):.6f}*{feat}")
    equation_str = "moisture% = " + " ".join(eq_terms)
    (out_dir / "regression_equation.txt").write_text(equation_str + "\n", encoding="utf-8")

    # Save held-out test predictions sorted by actual moisture (matching Fig. 6)
    eval_df = (
        pd.DataFrame({"name": name_test, "actual": y_test, "predicted": y_test_pred})
        .sort_values("actual")
        .reset_index(drop=True)
    )
    eval_df["abs_error"] = np.abs(eval_df["actual"] - eval_df["predicted"])
    eval_df["pct_error"] = eval_df["abs_error"] / eval_df["actual"] * 100.0
    eval_df.to_csv(out_dir / "test_predictions.csv", index=False)

    # Fig. 6: Actual vs. Predicted Moisture scatter plot
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=180)
    ax.scatter(eval_df.index, eval_df["actual"], label="Actual", s=28, alpha=0.9)
    ax.scatter(eval_df.index, eval_df["predicted"], label="Predicted", s=28, alpha=0.9)
    ax.set_title("Actual vs. Predicted Moisture")
    ax.set_xlabel("Held-out Test Sample Index (sorted by actual moisture)")
    ax.set_ylabel("Moisture (%)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig_path = out_dir / "fig6_actual_vs_predicted.png"
    fig.savefig(fig_path)
    plt.close(fig)

    # Resolve segment manifest for inference bundle
    if manifest_path is not None and manifest_path.exists():
        manifest_df = pd.read_csv(manifest_path)
        feature_specs = list(zip(manifest_df["column_name"], manifest_df["segment"], manifest_df["feature"]))
    else:
        paper_lookup = {feat: seg for seg, feat in PAPER_SELECTED_FEATURES}
        feature_specs = [(col, paper_lookup.get(col, "Green"), col) for col in feature_cols]

    # Save deployable model bundle
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "feature_specs": feature_specs,
        "extraction_mode": extraction_mode,
        "intercept_orig": intercept_orig,
        "beta_orig": beta_orig,
        "metrics": {
            "n_total": int(len(df)),
            "n_train": int(n_train),
            "n_test": int(len(y_test)),
            "n_features": int(p),
            "test_rmse": rmse,
            "test_mae": mae,
            "test_mape_pct": mape,
            "test_r2": r2,
            "train_r2": r2_train,
            "train_adj_r2": adj_r2_train,
            "anova_ss_reg": ss_reg,
            "anova_ss_res": ss_res,
            "anova_ss_tot": ss_tot,
            "anova_df_reg": df_reg,
            "anova_df_res": df_res,
            "anova_f_stat": f_stat,
            "anova_p_value": p_value,
            "standardized_cond_number": cond_number,
            "back_transform_max_diff": max_bt_diff,
            "random_state": random_state,
        },
    }
    joblib.dump(bundle, out_dir / "moisture_ols_model.joblib")
    (out_dir / "metrics_summary.json").write_text(json.dumps(bundle["metrics"], indent=2) + "\n", encoding="utf-8")

    # Pretty console summary
    print("=" * 68)
    print("IEEE iSAI-NLP 2025 — Multiple Linear Regression Results")
    print("=" * 68)
    print(f"Dataset          : {data_path.name} (N={len(df)}, Train={n_train}, Test={len(y_test)}, p={p})")
    print(f"Held-out RMSE    : {rmse:.3f} %   (Paper: 2.270)")
    print(f"Held-out MAE     : {mae:.3f} %   (Paper: 1.703)")
    print(f"Held-out MAPE    : {mape:.2f} %   (Paper: 9.75%)")
    print(f"Held-out R^2     : {r2:.3f}     (Paper: 0.840)")
    print(f"Train R^2 / Adj  : {r2_train:.3f} / {adj_r2_train:.3f}")
    print(f"ANOVA F({df_reg},{df_res})  : {f_stat:.3f}, p = {p_value:.2e}  (Paper: F(38,212)=51.026)")
    print(f"ANOVA SS         : Reg={ss_reg:.3f}, Res={ss_res:.3f}, Tot={ss_tot:.3f}  (Paper Tot: 8431.127)")
    print(f"Condition Number : {cond_number:.2e} (severe multicollinearity as noted in paper)")
    print(f"Saved outputs to : {out_dir}")
    print("=" * 68)

    return bundle["metrics"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate MLR model (paper Sec. IV.F–G & V.B–C).")
    parser.add_argument(
        "--data",
        type=Path,
        default=EXTRACTED_PAPER_DIR / "Correlated_FULL.xlsx",
        help="Path to Correlated_FULL.xlsx or Correlated_FULL.csv",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Optional selected_feature_manifest.csv")
    parser.add_argument("--out-dir", type=Path, default=OUTPUTS_DIR / "paper_296", help="Output directory")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    train_and_evaluate(
        data_path=args.data,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
        test_size=args.test_size,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()

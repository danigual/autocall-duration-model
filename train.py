"""Training script for autocall duration models.

Loads raw data, filters to executed RFQs, performs a chronological
train/test split, fits Ridge baseline + LightGBM + XGBoost pipelines,
reports evaluation metrics on the out-of-time test set, and serializes
the best-performing pipeline as artifacts/model.joblib.

Usage:
    python train.py
    python train.py --data-dir data --artifacts-dir artifacts
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from autocall.data.loader import load_daily_vol, load_reference, load_rfqs
from autocall.features.engineer import build_feature_pipeline
from autocall.models.baseline import build_baseline
from autocall.models.gbm import build_lgbm, build_xgb

TRAIN_CUTOFF = pd.Timestamp("2022-01-01")


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error.

    Args:
        y_true: Ground-truth target values. Must be strictly positive.
        y_pred: Model predictions.

    Returns:
        MAPE as a percentage (e.g. 18.5 means 18.5%).
    """
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def _evaluate(
    name: str,
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Compute and print regression metrics for one fitted model.

    Args:
        name: Human-readable model name for display.
        model: Fitted sklearn-compatible estimator.
        X_test: Test feature DataFrame (pre-transformed).
        y_test: Ground-truth target series.

    Returns:
        Dict with keys mae, rmse, r2, mape.
    """
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2   = r2_score(y_test, preds)
    mape = _mape(y_test.values, preds)

    print(
        f"  {name:<12}  MAE={mae:.2f}m  RMSE={rmse:.2f}m"
        f"  R2={r2:.4f}  MAPE={mape:.1f}%"
    )
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def main(data_dir: Path, artifacts_dir: Path) -> None:
    """Run the full training pipeline and export the best model artifact.

    Args:
        data_dir: Directory containing rfqs.csv, daily_volatility.csv,
            and underlyings_reference.csv.
        artifacts_dir: Directory where model.joblib will be written.
    """
    # --- Load raw data ---
    print("Loading data...")
    rfqs      = load_rfqs(data_dir)
    daily_vol = load_daily_vol(data_dir)
    reference = load_reference(data_dir)

    # --- Filter to executed RFQs with a known target ---
    data = rfqs[rfqs["executed"] & rfqs["avg_duration_months"].notna()].copy()
    print(f"  Executed RFQs with target: {len(data):,}")

    # --- Chronological train / test split ---
    train = data[data["requested_date"] < TRAIN_CUTOFF]
    test  = data[data["requested_date"] >= TRAIN_CUTOFF]
    print(f"  Train: {len(train):,} rows (before {TRAIN_CUTOFF.date()})")
    print(f"  Test:  {len(test):,} rows  ({TRAIN_CUTOFF.date()} onward)")

    X_train = train.drop(columns=["avg_duration_months"])
    y_train = train["avg_duration_months"]
    X_test  = test.drop(columns=["avg_duration_months"])
    y_test  = test["avg_duration_months"]

    # --- Pre-transform features once ---
    # VolatilityFeatures is expensive (explode + merge + groupby). Running it
    # inside RandomizedSearchCV would repeat it once per CV fold per iteration.
    # Pre-transforming avoids that and cuts training time dramatically.
    print("\nPre-computing features...")
    feature_pipe = build_feature_pipeline(daily_vol, reference)
    X_train_feat = feature_pipe.fit_transform(X_train)
    X_test_feat  = feature_pipe.transform(X_test)
    print(f"  Feature matrix: {X_train_feat.shape[1]} columns, {len(X_train_feat):,} train rows")

    # --- Train and evaluate model candidates on pre-transformed features ---
    print("\nTraining models...")
    candidates = {
        "ridge":    build_baseline(),
        "lightgbm": build_lgbm(),
        "xgboost":  build_xgb(),
    }

    fitted  = {}
    metrics = {}
    for name, model in candidates.items():
        print(f"  Fitting {name}...")
        model.fit(X_train_feat, y_train)
        fitted[name]  = model
        metrics[name] = _evaluate(name, model, X_test_feat, y_test)
        if hasattr(model, "best_params_"):
            print(f"    Best params: {model.best_params_}")

    # --- Select best model by MAE on the test set ---
    best_name  = min(metrics, key=lambda k: metrics[k]["mae"])
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name}  (test MAE = {metrics[best_name]['mae']:.2f} months)")

    # --- Build and serialize the full inference pipeline ---
    # The artifact must accept raw RFQ data, so we reassemble the full pipeline
    # using the pre-fitted feature steps and the best trained model.
    # Transformers are stateless so reusing their fitted instances is safe.
    artifact = Pipeline(feature_pipe.steps + [("model", best_model)])

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifacts_dir / "model.joblib"
    joblib.dump(artifact, artifact_path)
    print(f"Artifact saved to {artifact_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train autocall duration models and export the best artifact."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory containing the three source CSV files.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where model.joblib will be written.",
    )
    args = parser.parse_args()
    main(args.data_dir, args.artifacts_dir)

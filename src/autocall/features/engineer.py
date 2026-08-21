"""Pipeline assembly and feature selection.

Defines the canonical feature list and assembles all transformers into
a single sklearn Pipeline ready to be fitted by the training script.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from autocall.features.barriers import BarrierFeatures
from autocall.features.temporal import TemporalFeatures
from autocall.features.volatility import VolatilityFeatures

# Columns fed to the model. Redundant raw values (autocall_barrier_pct,
# protection_barrier_pct, no_call_period_months, quoted_implied_vol) are
# excluded because their information is captured by the derived features.
FEATURE_COLS: list[str] = [
    # contract structure
    "nominal_duration_months",
    "obs_freq_months",
    "obs_periods_total",
    "lockup_ratio",
    # barrier distances
    "autocall_barrier_distance",
    "protection_barrier_distance",
    "barrier_spread",
    # basket volatility
    "num_underlyings",
    "basket_rvol_min",
    "basket_rvol_mean",
    "basket_rvol_max",
    "basket_rvol_std",
    "basket_svol_min",
    "basket_svol_mean",
    "basket_svol_max",
    "num_sectors",
    "vol_spread",
    "vol_ratio",
    # categoricals — handled natively by LightGBM
    "product_type",
    "basket_type",
]

CATEGORICAL_COLS: list[str] = ["product_type", "basket_type"]


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects the model feature columns and casts categoricals.

    Acts as the final preprocessing step before the model. Drops every
    column not in the feature list and casts categorical columns to the
    pandas category dtype so LightGBM handles them natively.

    Args:
        columns: Ordered list of column names to retain.
        categorical_cols: Columns to cast to category dtype.
    """

    def __init__(
        self,
        columns: list[str] = FEATURE_COLS,
        categorical_cols: list[str] = CATEGORICAL_COLS,
    ) -> None:
        self.columns = columns
        self.categorical_cols = categorical_cols

    def fit(self, X: pd.DataFrame, y=None) -> "ColumnSelector":
        """No-op: column selection requires no fitting.

        Args:
            X: Input DataFrame (ignored).
            y: Target array (ignored).

        Returns:
            self
        """
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select feature columns and cast categoricals.

        Args:
            X: DataFrame containing all columns produced by the transformers.

        Returns:
            DataFrame with only the model feature columns, categoricals cast.
        """
        out = X[self.columns].copy()
        for col in self.categorical_cols:
            out[col] = out[col].astype("category")
        return out


def build_feature_pipeline(
    daily_vol: pd.DataFrame,
    reference: pd.DataFrame,
) -> Pipeline:
    """Assemble the feature engineering pipeline without a model step.

    Used in training to pre-transform features once before hyperparameter
    search, avoiding redundant transformer execution across CV folds.
    The returned pipeline is unfitted. Call fit_transform(X) on the
    training set and transform(X) on the test set.

    Args:
        daily_vol: Realized volatility table passed to VolatilityFeatures.
        reference: Underlyings reference table passed to VolatilityFeatures.

    Returns:
        Unfitted sklearn Pipeline with four steps:
        temporal -> barriers -> volatility -> selector.
    """
    return Pipeline([
        ("temporal",   TemporalFeatures()),
        ("barriers",   BarrierFeatures()),
        ("volatility", VolatilityFeatures(daily_vol, reference)),
        ("selector",   ColumnSelector()),
    ])


def build_pipeline(
    daily_vol: pd.DataFrame,
    reference: pd.DataFrame,
    model: BaseEstimator,
) -> Pipeline:
    """Assemble the full feature engineering and modelling pipeline.

    The returned pipeline is unfitted. Call pipeline.fit(X, y) to train.
    When serialized with joblib, the daily_vol and reference tables are
    embedded in the artifact so inference requires no external data files.

    Args:
        daily_vol: Realized volatility table passed to VolatilityFeatures.
        reference: Underlyings reference table passed to VolatilityFeatures.
        model: Any sklearn-compatible regressor (LGBMRegressor, Ridge, etc.).

    Returns:
        Unfitted sklearn Pipeline with five steps:
        temporal -> barriers -> volatility -> selector -> model.
    """
    return Pipeline([
        ("temporal",   TemporalFeatures()),
        ("barriers",   BarrierFeatures()),
        ("volatility", VolatilityFeatures(daily_vol, reference)),
        ("selector",   ColumnSelector()),
        ("model",      model),
    ])

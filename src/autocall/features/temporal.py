"""Temporal and contract-structure feature transformer.

Computes features derived exclusively from contract calendar fields
and observation cadence. No external data required.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TemporalFeatures(BaseEstimator, TransformerMixin):
    """Adds contract-structure features to the RFQ DataFrame.

    All transformations are stateless: fit() is a no-op and transform()
    is a pure function of the input columns. Compatible with sklearn Pipeline.

    Features produced:
        nominal_duration_months: Contractual life in months (end - start).
        obs_freq_months: Observation cadence normalized to months.
        obs_periods_total: Number of autocall observation windows in the contract.
        lockup_ratio: Fraction of contract life locked before first autocall window.
    """

    _DAYS_PER_MONTH: float = 365.25 / 12

    _FREQ_MAP: dict[str, float] = {
        # Daily
        "1D": 1 / 30,
        # Monthly
        "1M": 1.0,
        "M": 1.0,
        "Monthly": 1.0,
        "mensual": 1.0,
        "1 month": 1.0,
        # Bimonthly (undocumented variant found in EDA)
        "2M": 2.0,
        # Quarterly
        "3M": 3.0,
        "Q": 3.0,
        "Quarterly": 3.0,
        "trimestral": 3.0,
        "3 months": 3.0,
        # Semi-annual
        "6M": 6.0,
        # Annual
        "1Y": 12.0,
        "Y": 12.0,
        "12M": 12.0,
        "Annual": 12.0,
        "anual": 12.0,
    }

    def fit(self, X: pd.DataFrame, y=None) -> "TemporalFeatures":
        """No-op: this transformer has no learnable parameters.

        Args:
            X: Input DataFrame (ignored).
            y: Target array (ignored).

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute and append temporal features to X.

        Args:
            X: DataFrame containing at least start_date, end_date,
               observation_frequency, and no_call_period_months.

        Returns:
            Copy of X with four new columns appended.

        Raises:
            ValueError: If any observation_frequency value is not in _FREQ_MAP.
        """
        X = X.copy()

        X["nominal_duration_months"] = (
            (X["end_date"] - X["start_date"]).dt.days / self._DAYS_PER_MONTH
        )

        X["obs_freq_months"] = X["observation_frequency"].map(self._FREQ_MAP)

        unknown_mask = X["obs_freq_months"].isnull()
        if unknown_mask.any():
            bad_values = X.loc[unknown_mask, "observation_frequency"].unique().tolist()
            raise ValueError(
                f"Unknown observation_frequency values encountered: {bad_values}. "
                f"Add them to TemporalFeatures._FREQ_MAP."
            )

        X["obs_periods_total"] = X["nominal_duration_months"] / X["obs_freq_months"]

        X["lockup_ratio"] = (
            X["no_call_period_months"] / X["nominal_duration_months"]
        ).clip(lower=0.0, upper=1.0)

        return X

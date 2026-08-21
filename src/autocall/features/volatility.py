"""Volatility and basket composition feature transformer.

Joins each RFQ with realized and structural volatility data for every
underlying in the basket, then aggregates across basket components.
External data (daily_vol, reference) is passed at construction time so
the fitted pipeline can be serialized as a single joblib artifact.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class VolatilityFeatures(BaseEstimator, TransformerMixin):
    """Adds basket volatility and composition features to the RFQ DataFrame.

    Explodes the pipe-delimited underlyings column, joins each component
    with its realized and structural volatility, aggregates back to one
    row per RFQ, and computes implied-vs-realized spread features.

    For dates not present in daily_vol (inference on unseen dates), the
    transformer falls back to the latest available realized vol for each
    underlying so predictions remain possible without retraining.

    Args:
        daily_vol: DataFrame with columns [date, underlying, realized_vol_63d].
        reference: DataFrame with columns [underlying, sector, structural_base_vol].

    Features produced:
        num_underlyings: Number of assets in the basket.
        basket_rvol_min/mean/max/std: Realized vol aggregations across basket.
        basket_svol_min/mean/max: Structural vol aggregations across basket.
        num_sectors: Number of distinct sectors in the basket.
        vol_spread: quoted_implied_vol minus basket realized vol mean.
        vol_ratio: quoted_implied_vol divided by basket structural vol mean.
    """

    def __init__(self, daily_vol: pd.DataFrame, reference: pd.DataFrame) -> None:
        self.daily_vol = daily_vol
        self.reference = reference

    def fit(self, X: pd.DataFrame, y=None) -> "VolatilityFeatures":
        """No-op: external data is fixed, nothing is learned from training set.

        Args:
            X: Input DataFrame (ignored).
            y: Target array (ignored).

        Returns:
            self
        """
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute and append volatility and basket features to X.

        Args:
            X: DataFrame containing rfq_id, requested_date, underlyings
               (pipe-delimited), and quoted_implied_vol.

        Returns:
            Copy of X with volatility and basket features appended.
        """
        X = X.copy()

        # --- Step 1: explode pipe-delimited underlyings into one row per asset ---
        exploded = (
            X[["rfq_id", "requested_date", "underlyings"]]
            .copy()
            .assign(underlying=X["underlyings"].str.split("|"))
            .explode("underlying")
            .reset_index(drop=True)
        )

        # --- Step 2: join structural vol and sector from reference table ---
        exploded = exploded.merge(
            self.reference[["underlying", "sector", "structural_base_vol"]],
            on="underlying",
            how="left",
        )

        # --- Step 3: join realized vol by (underlying, requested_date) ---
        vol = self.daily_vol.rename(columns={"date": "requested_date"})
        exploded = exploded.merge(
            vol[["underlying", "requested_date", "realized_vol_63d"]],
            on=["underlying", "requested_date"],
            how="left",
        )

        # --- Step 4: fallback to latest available vol for unmatched dates ---
        missing = exploded["realized_vol_63d"].isnull()
        if missing.any():
            latest_vol = (
                self.daily_vol
                .sort_values("date")
                .groupby("underlying")["realized_vol_63d"]
                .last()
            )
            exploded.loc[missing, "realized_vol_63d"] = (
                exploded.loc[missing, "underlying"].map(latest_vol).values
            )

        # --- Step 5: aggregate from one-row-per-asset back to one-row-per-RFQ ---
        basket_size = (
            exploded.groupby("rfq_id")["underlying"]
            .count()
            .rename("num_underlyings")
        )

        rvol_agg = exploded.groupby("rfq_id")["realized_vol_63d"].agg(
            basket_rvol_min="min",
            basket_rvol_mean="mean",
            basket_rvol_max="max",
            basket_rvol_std="std",
        )

        svol_agg = exploded.groupby("rfq_id")["structural_base_vol"].agg(
            basket_svol_min="min",
            basket_svol_mean="mean",
            basket_svol_max="max",
        )

        sector_count = (
            exploded.groupby("rfq_id")["sector"]
            .nunique()
            .rename("num_sectors")
        )

        basket_features = pd.concat(
            [basket_size, rvol_agg, svol_agg, sector_count], axis=1
        )

        # --- Step 6: merge aggregated features back onto X ---
        X = X.merge(basket_features, on="rfq_id", how="left")

        # std is NaN for single-underlying baskets, no dispersion by definition
        X["basket_rvol_std"] = X["basket_rvol_std"].fillna(0.0)

        # --- Step 7: implied-vs-realized derived features ---
        X["vol_spread"] = X["quoted_implied_vol"] - X["basket_rvol_mean"]
        X["vol_ratio"] = X["quoted_implied_vol"] / X["basket_svol_mean"]

        return X

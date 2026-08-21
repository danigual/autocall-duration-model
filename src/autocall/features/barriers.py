"""Barrier distance feature transformer.

Computes features derived from the autocall and protection barrier levels.
Distances are more informative than raw barrier percentages because they
express how far the underlying must move relative to its initial level.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BarrierFeatures(BaseEstimator, TransformerMixin):
    """Adds barrier distance features to the RFQ DataFrame.

    All transformations are stateless: fit() is a no-op and transform()
    is a pure function of the input columns. Compatible with sklearn Pipeline.

    Features produced:
        autocall_barrier_distance: Distance the underlying must move above
            its initial level to trigger early redemption. Positive means
            the underlying must rally; negative means it can be below start.
        protection_barrier_distance: Downside buffer before the investor
            loses capital at maturity. Higher values mean more protection.
        barrier_spread: Gap between the autocall and protection barriers.
            Represents the operating range of the product.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "BarrierFeatures":
        """No-op: this transformer has no learnable parameters.

        Args:
            X: Input DataFrame (ignored).
            y: Target array (ignored).

        Returns:
            self
        """
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute and append barrier distance features to X.

        Args:
            X: DataFrame containing autocall_barrier_pct and
               protection_barrier_pct as float columns.

        Returns:
            Copy of X with three new columns appended.
        """
        X = X.copy()

        X["autocall_barrier_distance"] = X["autocall_barrier_pct"] - 1.0

        X["protection_barrier_distance"] = 1.0 - X["protection_barrier_pct"]

        X["barrier_spread"] = X["autocall_barrier_pct"] - X["protection_barrier_pct"]

        return X

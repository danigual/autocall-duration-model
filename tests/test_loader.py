"""Data contract tests for the three CSV loaders.

Each class validates one loader function. Tests check shape, dtypes,
null constraints, and value ranges — not business logic.
"""
from pathlib import Path

import pandas as pd
import pytest

from autocall.data.loader import load_daily_vol, load_reference, load_rfqs

DATA_DIR = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def rfqs():
    """Load rfqs.csv once for all tests in this module."""
    return load_rfqs(DATA_DIR)


@pytest.fixture(scope="module")
def daily_vol():
    """Load daily_volatility.csv once for all tests in this module."""
    return load_daily_vol(DATA_DIR)


@pytest.fixture(scope="module")
def reference():
    """Load underlyings_reference.csv once for all tests in this module."""
    return load_reference(DATA_DIR)


class TestLoadRfqs:
    """Validates the RFQ dataset contract after loading."""

    def test_shape(self, rfqs):
        """Dataset must have exactly 25,000 rows."""
        assert rfqs.shape[0] == 25_000

    def test_date_dtypes(self, rfqs):
        """Date columns must be datetime so temporal arithmetic works downstream."""
        for col in ("requested_date", "start_date", "end_date"):
            assert pd.api.types.is_datetime64_any_dtype(rfqs[col]), f"{col} is not datetime"

    def test_executed_is_bool(self, rfqs):
        """executed must be bool, not string 'True'/'False', for boolean indexing."""
        assert rfqs["executed"].dtype == bool

    def test_numeric_dtypes(self, rfqs):
        """Numeric columns must be float64 so the model can operate on them."""
        for col in (
            "autocall_barrier_pct",
            "protection_barrier_pct",
            "no_call_period_months",
            "quoted_implied_vol",
            "notional_credits",
        ):
            assert rfqs[col].dtype == float, f"{col} is not float"

    def test_no_nulls_in_key_columns(self, rfqs):
        """Columns required by feature engineering must have no missing values."""
        key_cols = [
            "rfq_id",
            "product_type",
            "basket_type",
            "underlyings",
            "autocall_barrier_pct",
            "protection_barrier_pct",
            "no_call_period_months",
            "observation_frequency",
            "quoted_implied_vol",
            "requested_date",
            "start_date",
            "end_date",
            "executed",
        ]
        nulls = rfqs[key_cols].isnull().sum()
        assert nulls.sum() == 0, f"Unexpected nulls:\n{nulls[nulls > 0]}"

    def test_barrier_ranges(self, rfqs):
        """Barriers are fractions of strike: autocall in (0,2), protection in (0,1)."""
        assert rfqs["autocall_barrier_pct"].between(0, 2).all()
        assert rfqs["protection_barrier_pct"].between(0, 1).all()

    def test_executed_rows_have_target(self, rfqs):
        """Executed RFQs must always have a non-null avg_duration_months label."""
        executed = rfqs[rfqs["executed"]]
        assert executed["avg_duration_months"].isnull().sum() == 0

    def test_unexecuted_rows_have_no_target(self, rfqs):
        """Unexecuted RFQs must never have a label — any value here would be leakage."""
        unexecuted = rfqs[~rfqs["executed"]]
        assert unexecuted["avg_duration_months"].isnull().all()


class TestLoadDailyVol:
    """Validates the daily volatility dataset contract after loading."""

    def test_shape(self, daily_vol):
        """Dataset must have exactly 44,744 rows."""
        assert daily_vol.shape[0] == 44_744

    def test_date_dtype(self, daily_vol):
        """date must be datetime for joining with requested_date."""
        assert pd.api.types.is_datetime64_any_dtype(daily_vol["date"])

    def test_vol_is_float(self, daily_vol):
        """Volatility must be float64 for numerical operations."""
        assert daily_vol["realized_vol_63d"].dtype == float

    def test_no_nulls(self, daily_vol):
        """No missing values allowed — a null vol would propagate silently into features."""
        assert daily_vol.isnull().sum().sum() == 0

    def test_vol_positive(self, daily_vol):
        """Realized volatility is always strictly positive by construction."""
        assert (daily_vol["realized_vol_63d"] > 0).all()


class TestLoadReference:
    """Validates the underlyings reference table contract after loading."""

    def test_shape(self, reference):
        """Reference table must have exactly 14 underlyings."""
        assert reference.shape[0] == 14

    def test_columns(self, reference):
        """Table must expose exactly the three expected columns."""
        assert set(reference.columns) == {"underlying", "sector", "structural_base_vol"}

    def test_structural_vol_is_float(self, reference):
        """Structural vol must be float64 for numerical operations."""
        assert reference["structural_base_vol"].dtype == float

    def test_no_nulls(self, reference):
        """No missing values — a null structural vol would break the vol ratio feature."""
        assert reference.isnull().sum().sum() == 0

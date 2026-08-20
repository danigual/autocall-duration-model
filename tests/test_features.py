"""Tests for feature transformers.

Uses synthetic DataFrames with known values so that results can be
asserted exactly rather than checking only structural properties.
"""
import pandas as pd
import pytest

from autocall.features.barriers import BarrierFeatures
from autocall.features.temporal import TemporalFeatures
from autocall.features.volatility import VolatilityFeatures


@pytest.fixture
def base_rfq() -> pd.DataFrame:
    """Minimal synthetic RFQ DataFrame for transformer tests.

    Two rows with controlled values so every output can be computed by hand.
    """
    return pd.DataFrame({
        "start_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "end_date":   pd.to_datetime(["2022-01-01", "2023-01-01"]),
        "observation_frequency": ["3M", "1Y"],
        "no_call_period_months": [6.0, 0.0],
    })


class TestTemporalFeatures:
    """Unit tests for the TemporalFeatures transformer."""

    def test_fit_returns_self(self, base_rfq):
        """fit() must return self so Pipeline chaining works."""
        tf = TemporalFeatures()
        assert tf.fit(base_rfq) is tf

    def test_does_not_mutate_input(self, base_rfq):
        """transform() must not add columns to the original DataFrame."""
        original_cols = set(base_rfq.columns)
        TemporalFeatures().fit_transform(base_rfq)
        assert set(base_rfq.columns) == original_cols

    def test_nominal_duration_months(self, base_rfq):
        """nominal_duration_months = (end - start).days / (365.25 / 12)."""
        out = TemporalFeatures().fit_transform(base_rfq)
        days_per_month = 365.25 / 12
        expected_0 = (pd.Timestamp("2022-01-01") - pd.Timestamp("2020-01-01")).days / days_per_month
        expected_1 = (pd.Timestamp("2023-01-01") - pd.Timestamp("2020-01-01")).days / days_per_month
        assert out["nominal_duration_months"].iloc[0] == pytest.approx(expected_0, rel=1e-6)
        assert out["nominal_duration_months"].iloc[1] == pytest.approx(expected_1, rel=1e-6)

    def test_obs_freq_months_mapping(self, base_rfq):
        """observation_frequency strings must map to the correct numeric cadence."""
        out = TemporalFeatures().fit_transform(base_rfq)
        assert out["obs_freq_months"].iloc[0] == pytest.approx(3.0)
        assert out["obs_freq_months"].iloc[1] == pytest.approx(12.0)

    def test_all_freq_variants_are_mapped(self):
        """Every observation_frequency variant found in the dataset must be in _FREQ_MAP."""
        known_variants = [
            "1D",
            "1M", "M", "Monthly", "mensual", "1 month",
            "2M",
            "3M", "Q", "Quarterly", "trimestral", "3 months",
            "6M",
            "1Y", "Y", "12M", "Annual", "anual",
        ]
        for variant in known_variants:
            assert variant in TemporalFeatures._FREQ_MAP, f"Missing variant: {variant}"

    def test_obs_periods_total(self, base_rfq):
        """obs_periods_total = nominal_duration_months / obs_freq_months."""
        out = TemporalFeatures().fit_transform(base_rfq)
        expected = out["nominal_duration_months"] / out["obs_freq_months"]
        pd.testing.assert_series_equal(out["obs_periods_total"], expected, check_names=False)

    def test_lockup_ratio_basic(self, base_rfq):
        """lockup_ratio = no_call_period_months / nominal_duration_months."""
        out = TemporalFeatures().fit_transform(base_rfq)
        assert out["lockup_ratio"].iloc[0] == pytest.approx(
            6.0 / out["nominal_duration_months"].iloc[0], rel=1e-6
        )

    def test_lockup_ratio_zero(self, base_rfq):
        """A zero lockup period must produce a lockup_ratio of exactly 0."""
        out = TemporalFeatures().fit_transform(base_rfq)
        assert out["lockup_ratio"].iloc[1] == pytest.approx(0.0)

    def test_lockup_ratio_clipped_at_one(self):
        """lockup_ratio must be clipped to 1.0 when no_call exceeds nominal duration."""
        df = pd.DataFrame({
            "start_date": pd.to_datetime(["2020-01-01"]),
            "end_date":   pd.to_datetime(["2022-01-01"]),
            "observation_frequency": ["1M"],
            "no_call_period_months": [999.0],
        })
        out = TemporalFeatures().fit_transform(df)
        assert out["lockup_ratio"].iloc[0] == pytest.approx(1.0)

    def test_unknown_frequency_raises(self):
        """An unrecognized observation_frequency must raise ValueError immediately."""
        df = pd.DataFrame({
            "start_date": pd.to_datetime(["2020-01-01"]),
            "end_date":   pd.to_datetime(["2022-01-01"]),
            "observation_frequency": ["UNKNOWN_FREQ"],
            "no_call_period_months": [0.0],
        })
        with pytest.raises(ValueError, match="Unknown observation_frequency"):
            TemporalFeatures().fit_transform(df)

    def test_no_nulls_in_output(self, base_rfq):
        """No NaN values must appear in any of the four new columns."""
        out = TemporalFeatures().fit_transform(base_rfq)
        new_cols = ["nominal_duration_months", "obs_freq_months", "obs_periods_total", "lockup_ratio"]
        assert out[new_cols].isnull().sum().sum() == 0


@pytest.fixture
def barrier_rfq() -> pd.DataFrame:
    """Synthetic RFQ DataFrame with controlled barrier values for exact assertions."""
    return pd.DataFrame({
        "autocall_barrier_pct":    [1.05, 0.95],
        "protection_barrier_pct":  [0.70, 0.60],
    })


class TestBarrierFeatures:
    """Unit tests for the BarrierFeatures transformer."""

    def test_fit_returns_self(self, barrier_rfq):
        """fit() must return self so Pipeline chaining works."""
        bf = BarrierFeatures()
        assert bf.fit(barrier_rfq) is bf

    def test_does_not_mutate_input(self, barrier_rfq):
        """transform() must not add columns to the original DataFrame."""
        original_cols = set(barrier_rfq.columns)
        BarrierFeatures().fit_transform(barrier_rfq)
        assert set(barrier_rfq.columns) == original_cols

    def test_autocall_barrier_distance(self, barrier_rfq):
        """autocall_barrier_distance = autocall_barrier_pct - 1.0."""
        out = BarrierFeatures().fit_transform(barrier_rfq)
        assert out["autocall_barrier_distance"].iloc[0] == pytest.approx(0.05)
        assert out["autocall_barrier_distance"].iloc[1] == pytest.approx(-0.05)

    def test_protection_barrier_distance(self, barrier_rfq):
        """protection_barrier_distance = 1.0 - protection_barrier_pct."""
        out = BarrierFeatures().fit_transform(barrier_rfq)
        assert out["protection_barrier_distance"].iloc[0] == pytest.approx(0.30)
        assert out["protection_barrier_distance"].iloc[1] == pytest.approx(0.40)

    def test_barrier_spread(self, barrier_rfq):
        """barrier_spread = autocall_barrier_pct - protection_barrier_pct."""
        out = BarrierFeatures().fit_transform(barrier_rfq)
        assert out["barrier_spread"].iloc[0] == pytest.approx(1.05 - 0.70)
        assert out["barrier_spread"].iloc[1] == pytest.approx(0.95 - 0.60)

    def test_no_nulls_in_output(self, barrier_rfq):
        """No NaN values must appear in any of the three new columns."""
        out = BarrierFeatures().fit_transform(barrier_rfq)
        new_cols = ["autocall_barrier_distance", "protection_barrier_distance", "barrier_spread"]
        assert out[new_cols].isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# VolatilityFeatures fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ref_table() -> pd.DataFrame:
    """Synthetic reference table with three underlyings across two sectors."""
    return pd.DataFrame({
        "underlying":          ["AAAA", "BBBB", "CCCC"],
        "sector":              ["Energy", "Tech", "Energy"],
        "structural_base_vol": [0.30,    0.40,   0.20],
    })


@pytest.fixture
def vol_table() -> pd.DataFrame:
    """Synthetic daily vol table. AAAA has two dates to test the fallback."""
    return pd.DataFrame({
        "date":             pd.to_datetime(["2020-01-10", "2020-01-15", "2020-01-15", "2020-01-15"]),
        "underlying":       ["AAAA",        "AAAA",        "BBBB",        "CCCC"],
        "realized_vol_63d": [0.32,          0.35,          0.45,          0.25],
    })


@pytest.fixture
def vol_rfq() -> pd.DataFrame:
    """Three synthetic RFQs: one multi-asset, one single-asset, one with unseen date."""
    return pd.DataFrame({
        "rfq_id":           ["R01",         "R02",         "R03"],
        "requested_date":   pd.to_datetime(["2020-01-15", "2020-01-15", "2020-01-20"]),
        "underlyings":      ["AAAA|BBBB",   "CCCC",        "AAAA"],
        "quoted_implied_vol": [0.50,         0.30,          0.40],
    })


class TestVolatilityFeatures:
    """Unit tests for the VolatilityFeatures transformer."""

    def test_fit_returns_self(self, vol_rfq, vol_table, ref_table):
        """fit() must return self so Pipeline chaining works."""
        vf = VolatilityFeatures(vol_table, ref_table)
        assert vf.fit(vol_rfq) is vf

    def test_does_not_mutate_input(self, vol_rfq, vol_table, ref_table):
        """transform() must not add columns to the original DataFrame."""
        original_cols = set(vol_rfq.columns)
        VolatilityFeatures(vol_table, ref_table).fit_transform(vol_rfq)
        assert set(vol_rfq.columns) == original_cols

    def test_num_underlyings_multi(self, vol_rfq, vol_table, ref_table):
        """R01 has two underlyings — AAAA and BBBB."""
        out = VolatilityFeatures(vol_table, ref_table).fit_transform(vol_rfq)
        assert out.loc[out["rfq_id"] == "R01", "num_underlyings"].iloc[0] == 2

    def test_num_underlyings_single(self, vol_rfq, vol_table, ref_table):
        """R02 has one underlying — CCCC."""
        out = VolatilityFeatures(vol_table, ref_table).fit_transform(vol_rfq)
        assert out.loc[out["rfq_id"] == "R02", "num_underlyings"].iloc[0] == 1

    def test_basket_rvol_mean(self, vol_rfq, vol_table, ref_table):
        """R01 rvol_mean must be the average of AAAA (0.35) and BBBB (0.45)."""
        out = VolatilityFeatures(vol_table, ref_table).fit_transform(vol_rfq)
        assert out.loc[out["rfq_id"] == "R01", "basket_rvol_mean"].iloc[0] == pytest.approx(0.40)

    def test_basket_rvol_std_single(self, vol_rfq, vol_table, ref_table):
        """Single-underlying basket must have rvol_std of 0, not NaN."""
        out = VolatilityFeatures(vol_table, ref_table).fit_transform(vol_rfq)
        assert out.loc[out["rfq_id"] == "R02", "basket_rvol_std"].iloc[0] == pytest.approx(0.0)

    def test_num_sectors(self, vol_rfq, vol_table, ref_table):
        """R01 has AAAA (Energy) and BBBB (Tech) — two distinct sectors."""
        out = VolatilityFeatures(vol_table, ref_table).fit_transform(vol_rfq)
        assert out.loc[out["rfq_id"] == "R01", "num_sectors"].iloc[0] == 2

    def test_num_sectors_single(self, vol_rfq, vol_table, ref_table):
        """R02 has only CCCC (Energy) — one sector."""
        out = VolatilityFeatures(vol_table, ref_table).fit_transform(vol_rfq)
        assert out.loc[out["rfq_id"] == "R02", "num_sectors"].iloc[0] == 1

    def test_vol_spread(self, vol_rfq, vol_table, ref_table):
        """vol_spread = quoted_implied_vol - basket_rvol_mean."""
        out = VolatilityFeatures(vol_table, ref_table).fit_transform(vol_rfq)
        row = out.loc[out["rfq_id"] == "R01"].iloc[0]
        assert row["vol_spread"] == pytest.approx(row["quoted_implied_vol"] - row["basket_rvol_mean"])

    def test_vol_ratio(self, vol_rfq, vol_table, ref_table):
        """vol_ratio = quoted_implied_vol / basket_svol_mean."""
        out = VolatilityFeatures(vol_table, ref_table).fit_transform(vol_rfq)
        row = out.loc[out["rfq_id"] == "R01"].iloc[0]
        assert row["vol_ratio"] == pytest.approx(row["quoted_implied_vol"] / row["basket_svol_mean"])

    def test_fallback_for_unseen_date(self, vol_rfq, vol_table, ref_table):
        """R03 uses date 2020-01-20 not in vol_table — must fall back to latest AAAA vol (0.35)."""
        out = VolatilityFeatures(vol_table, ref_table).fit_transform(vol_rfq)
        assert out.loc[out["rfq_id"] == "R03", "basket_rvol_mean"].iloc[0] == pytest.approx(0.35)

    def test_no_nulls_in_output(self, vol_rfq, vol_table, ref_table):
        """No NaN values must appear in any of the new volatility columns."""
        out = VolatilityFeatures(vol_table, ref_table).fit_transform(vol_rfq)
        new_cols = [
            "num_underlyings",
            "basket_rvol_min", "basket_rvol_mean", "basket_rvol_max", "basket_rvol_std",
            "basket_svol_min", "basket_svol_mean", "basket_svol_max",
            "num_sectors", "vol_spread", "vol_ratio",
        ]
        assert out[new_cols].isnull().sum().sum() == 0

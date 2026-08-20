from pathlib import Path

import pandas as pd


def load_rfqs(data_dir: Path) -> pd.DataFrame:
    """Load and type-cast the RFQ dataset.

    Parses date columns explicitly and enforces numeric dtypes so that
    downstream modules always receive a well-typed DataFrame regardless
    of how the CSV was originally encoded.

    Args:
        data_dir: Directory that contains rfqs.csv.

    Returns:
        DataFrame with one row per RFQ and the following guaranteed dtypes:
        - requested_date, start_date, end_date: datetime64[ns]
        - executed: bool
        - autocall_barrier_pct, protection_barrier_pct,
          no_call_period_months, quoted_implied_vol,
          notional_credits: float64
        - product_type, basket_type, underlyings,
          observation_frequency, counterparty, trader_id: object (str)
    """
    df = pd.read_csv(
        data_dir / "rfqs.csv",
        parse_dates=["requested_date", "start_date", "end_date"],
    )
    df["executed"] = df["executed"].astype(bool)
    for col in (
        "autocall_barrier_pct",
        "protection_barrier_pct",
        "no_call_period_months",
        "quoted_implied_vol",
        "notional_credits",
    ):
        df[col] = df[col].astype(float)
    return df


def load_daily_vol(data_dir: Path) -> pd.DataFrame:
    """Load and type-cast the daily realized volatility dataset.

    Args:
        data_dir: Directory that contains daily_volatility.csv.

    Returns:
        DataFrame with one row per (date, underlying) pair and the
        following guaranteed dtypes:
        - date: datetime64[ns]
        - underlying: object (str)
        - realized_vol_63d: float64
    """
    df = pd.read_csv(
        data_dir / "daily_volatility.csv",
        parse_dates=["date"],
    )
    df["realized_vol_63d"] = df["realized_vol_63d"].astype(float)
    return df


def load_reference(data_dir: Path) -> pd.DataFrame:
    """Load the underlyings reference table.

    Args:
        data_dir: Directory that contains underlyings_reference.csv.

    Returns:
        DataFrame with one row per underlying and the following
        guaranteed dtypes:
        - underlying: object (str)
        - sector: object (str)
        - structural_base_vol: float64
    """
    df = pd.read_csv(data_dir / "underlyings_reference.csv")
    df["structural_base_vol"] = df["structural_base_vol"].astype(float)
    return df

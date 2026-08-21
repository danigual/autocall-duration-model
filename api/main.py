"""FastAPI inference server for autocall duration prediction.

Loads the trained pipeline artifact once at startup and exposes a single
POST /predict endpoint. The model artifact is a full sklearn Pipeline that
accepts raw RFQ data and returns a predicted avg_duration_months.

Usage:
    uvicorn api.main:app --reload
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model artifact once at server startup and release on shutdown."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run train.py first to generate the artifact."
        )
    app.state.model = joblib.load(MODEL_PATH)
    yield
    del app.state.model


app = FastAPI(
    title="Autocall Duration API",
    description="Predicts the expected lifespan of autocallable structured notes.",
    version="0.1.0",
    lifespan=lifespan,
)


class RFQInput(BaseModel):
    """Input schema for a single RFQ prediction request.

    All fields must be provided at the time of quoting (no look-ahead).
    Dates must be in ISO 8601 format (YYYY-MM-DD).
    """

    product_type: str = Field(..., description="Structured payoff category.")
    basket_type: str = Field(..., description="'worst_of' or 'single'.")
    underlyings: str = Field(..., description="Pipe-delimited tickers, e.g. 'AAPL|GOOG'.")
    autocall_barrier_pct: float = Field(..., description="Early redemption trigger level (fraction of strike).")
    protection_barrier_pct: float = Field(..., description="Capital downside barrier at maturity.")
    no_call_period_months: float = Field(..., description="Lockup period before first autocall observation.")
    observation_frequency: str = Field(..., description="Barrier check cadence, e.g. '3M', 'Monthly'.")
    quoted_implied_vol: float = Field(..., description="Implied volatility quoted by the desk.")
    requested_date: str = Field(..., description="Date of RFQ pricing (YYYY-MM-DD).")
    start_date: str = Field(..., description="Contract start date (YYYY-MM-DD).")
    end_date: str = Field(..., description="Contract end date (YYYY-MM-DD).")


class PredictionOutput(BaseModel):
    """Output schema for the duration prediction."""

    avg_duration_months: float = Field(
        ..., description="Predicted expected lifespan in months."
    )


@app.get("/health")
def health(request: Request) -> dict:
    """Confirm that the server is running and the model is loaded."""
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "model")}


@app.post("/predict", response_model=PredictionOutput)
def predict(rfq: RFQInput, request: Request) -> PredictionOutput:
    """Predict the expected duration of an autocallable note at RFQ time.

    Accepts a single RFQ payload, runs it through the full feature engineering
    pipeline, and returns the predicted avg_duration_months.

    Args:
        rfq: Validated RFQ input fields.
        request: FastAPI request object used to access the loaded model.

    Returns:
        PredictionOutput with the predicted duration in months.

    Raises:
        HTTPException 500: If the model raises an unexpected error.
    """
    row = rfq.model_dump()

    # VolatilityFeatures requires rfq_id for the basket groupby aggregation.
    row["rfq_id"] = "INF_001"

    # Convert date strings to pandas Timestamps as expected by TemporalFeatures.
    for date_col in ("requested_date", "start_date", "end_date"):
        row[date_col] = pd.Timestamp(row[date_col])

    df = pd.DataFrame([row])

    try:
        raw_pred = request.app.state.model.predict(df)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Clip to [0, nominal_duration]: a product cannot outlive its contractual
    # life at inference time (extensions are unknown at RFQ date).
    days_per_month = 365.25 / 12
    nominal = (pd.Timestamp(rfq.end_date) - pd.Timestamp(rfq.start_date)).days / days_per_month
    prediction = float(max(0.0, min(raw_pred, nominal)))

    return PredictionOutput(avg_duration_months=prediction)

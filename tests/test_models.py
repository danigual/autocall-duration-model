"""Tests for model factory functions.

Covers structural properties (correct types and step names) and smoke
tests to verify that each model fits and predicts on numeric data without
errors. At this stage the pipeline's ColumnSelector has already run, so
the model receives a plain numeric DataFrame with no categoricals.
"""
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from autocall.models.baseline import build_baseline
from autocall.models.gbm import build_lgbm, build_xgb


@pytest.fixture
def numeric_data() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic numeric training set with three features and 50 samples.

    Values are drawn from a standard normal distribution. The target is a
    positive continuous variable (avg_duration_months is always positive).
    """
    rng = np.random.default_rng(seed=42)
    X = pd.DataFrame({
        "nominal_duration_months": rng.uniform(6, 36, size=50),
        "vol_spread":              rng.normal(0, 0.1, size=50),
        "lockup_ratio":            rng.uniform(0, 1, size=50),
    })
    y = pd.Series(rng.uniform(1, 36, size=50), name="avg_duration_months")
    return X, y


class TestBuildBaseline:
    """Unit tests for the build_baseline factory function."""

    def test_returns_pipeline(self):
        """build_baseline() must return an sklearn Pipeline."""
        assert isinstance(build_baseline(), Pipeline)

    def test_pipeline_has_two_steps(self):
        """Pipeline must contain exactly two steps: scaler and ridge."""
        pipe = build_baseline()
        assert len(pipe.steps) == 2

    def test_first_step_is_standard_scaler(self):
        """First step must be named 'scaler' and be a StandardScaler."""
        pipe = build_baseline()
        name, estimator = pipe.steps[0]
        assert name == "scaler"
        assert isinstance(estimator, StandardScaler)

    def test_second_step_is_ridge(self):
        """Second step must be named 'ridge' and be a Ridge estimator."""
        pipe = build_baseline()
        name, estimator = pipe.steps[1]
        assert name == "ridge"
        assert isinstance(estimator, Ridge)

    def test_alpha_is_forwarded(self):
        """alpha parameter must be forwarded to the Ridge estimator."""
        pipe = build_baseline(alpha=10.0)
        assert pipe.named_steps["ridge"].alpha == pytest.approx(10.0)

    def test_fits_without_error(self, numeric_data):
        """Pipeline must fit on a numeric DataFrame without raising."""
        X, y = numeric_data
        build_baseline().fit(X, y)

    def test_predict_returns_array(self, numeric_data):
        """predict() must return a numpy array of the same length as X."""
        X, y = numeric_data
        pipe = build_baseline().fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(X)

    def test_predictions_are_finite(self, numeric_data):
        """All predictions must be finite real numbers, no NaN or Inf."""
        X, y = numeric_data
        pipe = build_baseline().fit(X, y)
        preds = pipe.predict(X)
        assert np.all(np.isfinite(preds))


class TestBuildLgbm:
    """Unit tests for the build_lgbm factory function."""

    def test_returns_randomized_search_cv(self):
        """build_lgbm() must return a RandomizedSearchCV instance."""
        assert isinstance(build_lgbm(), RandomizedSearchCV)

    def test_cv_is_time_series_split(self):
        """CV strategy must be TimeSeriesSplit to respect temporal ordering."""
        assert isinstance(build_lgbm().cv, TimeSeriesSplit)

    def test_n_splits_forwarded(self):
        """n_splits must be forwarded to the TimeSeriesSplit inside the CV."""
        search = build_lgbm(n_splits=3)
        assert search.cv.n_splits == 3

    def test_scoring_is_mae(self):
        """scoring must be neg_mean_absolute_error."""
        assert build_lgbm().scoring == "neg_mean_absolute_error"

    def test_refit_is_true(self):
        """refit=True so the best model is retained after search."""
        assert build_lgbm().refit is True

    def test_fits_and_predicts(self, numeric_data):
        """Search must complete and produce predictions on the training set."""
        X, y = numeric_data
        search = build_lgbm(n_iter=2, n_splits=2).fit(X, y)
        preds = search.predict(X)
        assert len(preds) == len(X)

    def test_predictions_are_finite(self, numeric_data):
        """All predictions must be finite real numbers, no NaN or Inf."""
        X, y = numeric_data
        search = build_lgbm(n_iter=2, n_splits=2).fit(X, y)
        assert np.all(np.isfinite(search.predict(X)))

    def test_best_estimator_is_lgbm(self, numeric_data):
        """After fitting, best_estimator_ must be a LGBMRegressor."""
        X, y = numeric_data
        search = build_lgbm(n_iter=2, n_splits=2).fit(X, y)
        assert isinstance(search.best_estimator_, LGBMRegressor)


class TestBuildXgb:
    """Unit tests for the build_xgb factory function."""

    def test_returns_randomized_search_cv(self):
        """build_xgb() must return a RandomizedSearchCV instance."""
        assert isinstance(build_xgb(), RandomizedSearchCV)

    def test_cv_is_time_series_split(self):
        """CV strategy must be TimeSeriesSplit to respect temporal ordering."""
        assert isinstance(build_xgb().cv, TimeSeriesSplit)

    def test_n_splits_forwarded(self):
        """n_splits must be forwarded to the TimeSeriesSplit inside the CV."""
        search = build_xgb(n_splits=3)
        assert search.cv.n_splits == 3

    def test_scoring_is_mae(self):
        """scoring must be neg_mean_absolute_error."""
        assert build_xgb().scoring == "neg_mean_absolute_error"

    def test_refit_is_true(self):
        """refit=True so the best model is retained after search."""
        assert build_xgb().refit is True

    def test_fits_and_predicts(self, numeric_data):
        """Search must complete and produce predictions on the training set."""
        X, y = numeric_data
        search = build_xgb(n_iter=2, n_splits=2).fit(X, y)
        preds = search.predict(X)
        assert len(preds) == len(X)

    def test_predictions_are_finite(self, numeric_data):
        """All predictions must be finite real numbers, no NaN or Inf."""
        X, y = numeric_data
        search = build_xgb(n_iter=2, n_splits=2).fit(X, y)
        assert np.all(np.isfinite(search.predict(X)))

    def test_best_estimator_is_xgb(self, numeric_data):
        """After fitting, best_estimator_ must be a XGBRegressor."""
        X, y = numeric_data
        search = build_xgb(n_iter=2, n_splits=2).fit(X, y)
        assert isinstance(search.best_estimator_, XGBRegressor)

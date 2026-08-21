"""Gradient boosting model factories with hyperparameter search.

Each function returns a RandomizedSearchCV wrapping either LGBMRegressor or
XGBRegressor. Hyperparameters are sampled from continuous and discrete
distributions rather than a fixed grid, which is more efficient when the
search space is large (Bergstra & Bengio, 2012).

TimeSeriesSplit is used as the CV strategy so that within the training window
each validation fold always lies after its training fold in calendar time,
preventing market-regime leakage during hyperparameter selection.
"""
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def build_lgbm(
    n_iter: int = 20,
    n_splits: int = 3,
    random_state: int = 42,
) -> RandomizedSearchCV:
    """Build a LightGBM regressor with randomized hyperparameter search.

    LightGBM grows trees leaf-wise, which tends to produce lower training
    error faster than level-wise growth (XGBoost default) on tabular data.
    Categorical columns (product_type, basket_type) passed as pandas category
    dtype are handled natively without one-hot encoding.

    Args:
        n_iter: Number of random hyperparameter combinations to evaluate.
        n_splits: Number of temporal folds in TimeSeriesSplit.
        random_state: Seed for reproducibility of both the search and the model.

    Returns:
        Unfitted RandomizedSearchCV. Calling .fit() triggers the full search
        and stores the best estimator in .best_estimator_.
    """
    estimator = LGBMRegressor(
        random_state=random_state,
        verbose=-1,
    )

    param_dist = {
        "n_estimators":      randint(100, 401),
        "learning_rate":     loguniform(0.01, 0.3),
        "num_leaves":        randint(15, 128),
        "min_child_samples": randint(20, 100),
        "subsample":         uniform(0.6, 0.4),
        "colsample_bytree":  uniform(0.6, 0.4),
    }

    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=1,
        refit=True,
        verbose=1,
    )


def build_xgb(
    n_iter: int = 20,
    n_splits: int = 3,
    random_state: int = 42,
) -> RandomizedSearchCV:
    """Build an XGBoost regressor with randomized hyperparameter search.

    XGBoost grows trees level-wise with native L2 regularisation (reg_lambda).
    enable_categorical=True allows XGBoost to consume pandas category dtype
    columns directly, matching the output of ColumnSelector.

    Args:
        n_iter: Number of random hyperparameter combinations to evaluate.
        n_splits: Number of temporal folds in TimeSeriesSplit.
        random_state: Seed for reproducibility of both the search and the model.

    Returns:
        Unfitted RandomizedSearchCV. Calling .fit() triggers the full search
        and stores the best estimator in .best_estimator_.
    """
    estimator = XGBRegressor(
        random_state=random_state,
        verbosity=0,
        enable_categorical=True,
    )

    param_dist = {
        "n_estimators":     randint(100, 401),
        "learning_rate":    loguniform(0.01, 0.3),
        "max_depth":        randint(3, 10),
        "min_child_weight": randint(1, 20),
        "subsample":        uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_lambda":       loguniform(0.1, 10.0),
    }

    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_mean_absolute_error",
        random_state=random_state,
        n_jobs=1,
        refit=True,
        verbose=1,
    )

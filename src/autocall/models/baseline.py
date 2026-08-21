"""Ridge regression baseline model.

Provides a simple linear baseline to benchmark against gradient boosting.
The pipeline wraps StandardScaler + Ridge so the returned object is a
single sklearn-compatible estimator that can be passed to build_pipeline().
"""
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_baseline(alpha: float = 1.0) -> Pipeline:
    """Build a Ridge regression baseline with feature scaling.

    Features must be scaled before Ridge because its L2 penalty is
    sensitive to the magnitude of coefficients, which depends on the
    scale of each input feature. StandardScaler normalises every column
    to zero mean and unit variance before the Ridge step sees the data.

    The returned pipeline is unfitted. Call pipeline.fit(X, y) to train.
    It is designed to be passed as the model argument to build_pipeline().

    Args:
        alpha: L2 regularisation strength. Higher values shrink coefficients
            more aggressively. Defaults to 1.0 (sklearn default).

    Returns:
        Unfitted Pipeline with two steps: scaler -> ridge.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=alpha)),
    ])

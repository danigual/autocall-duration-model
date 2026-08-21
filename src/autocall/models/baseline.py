"""Ridge regression baseline model.

Provides a simple linear baseline to benchmark against gradient boosting.
Numeric features are scaled with StandardScaler; categorical features
(product_type, basket_type) are one-hot encoded. Column types are detected
automatically via make_column_selector so no column names are hardcoded here.
"""
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_baseline(alpha: float = 1.0) -> Pipeline:
    """Build a Ridge regression baseline with type-aware preprocessing.

    StandardScaler requires numeric inputs and cannot handle pandas category
    columns. A ColumnTransformer is used to route numeric columns through
    StandardScaler and categorical columns through OneHotEncoder before
    Ridge sees any data.

    make_column_selector detects column types from pandas dtypes at fit time,
    so no column names need to be hardcoded in this function.

    The returned pipeline is unfitted. Call pipeline.fit(X, y) to train.
    It is designed to be passed as the model argument to build_pipeline().

    Args:
        alpha: L2 regularisation strength. Higher values shrink coefficients
            more aggressively. Defaults to 1.0 (sklearn default).

    Returns:
        Unfitted Pipeline with two steps: preprocessor -> ridge.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                make_column_selector(dtype_exclude="category"),
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                make_column_selector(dtype_include="category"),
            ),
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("ridge",        Ridge(alpha=alpha)),
    ])

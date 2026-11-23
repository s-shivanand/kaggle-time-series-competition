"""Read reported Kaggle scores and verify submission artifact structure."""

from pathlib import Path

import numpy as np
import pandas as pd

SUBMISSION_COLUMNS = ["id", "unit_sales"]


def read_kaggle_scores(path: Path) -> pd.DataFrame:
    """Read the transcribed scores with their provenance in display order."""
    return pd.read_csv(path)


def validate_id_order(ids: pd.Series, expected_ids: pd.Series) -> None:
    """Require each test ID to be unique and in sample submission order."""
    if ids.empty or ids.hasnans or not ids.is_unique:
        raise ValueError("Submission IDs must be nonempty, unique, and nonmissing")
    if not np.array_equal(ids.to_numpy(), expected_ids.to_numpy()):
        raise ValueError("Submission IDs must match the sample submission in row order")


def validate_submission(frame: pd.DataFrame, expected_ids: pd.Series) -> None:
    """Check headers, exact ID order, and finite, nonnegative predictions."""
    if list(frame.columns) != SUBMISSION_COLUMNS:
        raise ValueError("Submission columns must be id,unit_sales in that order")
    validate_id_order(frame["id"], expected_ids)
    sales = frame["unit_sales"]
    if not pd.api.types.is_numeric_dtype(sales) or not (
        np.isfinite(sales).all() and (sales >= 0).all() and not sales.hasnans
    ):
        raise ValueError("unit_sales must be numeric, finite, and nonnegative")

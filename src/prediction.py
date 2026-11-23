"""Predict and export the basic notebook submissions with ID checks."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.evaluation import validate_id_order, validate_submission
from src.preprocessing import prepare_test_data


def predict_submission(
    pipeline: Pipeline, test: pd.DataFrame, sample: pd.DataFrame
) -> pd.DataFrame:
    """Predict in raw test order, clip at zero, and associate each row with its ID."""
    ids = test["id"]
    expected_ids = sample["id"]
    validate_id_order(ids, expected_ids)
    predictions = np.clip(pipeline.predict(prepare_test_data(test)), 0, None)
    submission = pd.DataFrame({"id": ids.to_numpy(), "unit_sales": predictions})
    validate_submission(submission, expected_ids)
    return submission


def predict_rolling_submission(
    pipeline: Pipeline, features: pd.DataFrame, sample: pd.DataFrame
) -> pd.DataFrame:
    """Apply the historical inverse transform and positional template assignment."""
    result = sample.copy()
    result["unit_sales"] = pipeline.predict(features)
    result["unit_sales"] = np.maximum(0, result["unit_sales"])
    result["unit_sales"] = np.expm1(result["unit_sales"])
    return result


def write_submission(
    frame: pd.DataFrame, expected_ids: pd.Series, output: Path
) -> None:
    """Publish a complete CSV at a new path without overwriting existing results."""
    validate_submission(frame, expected_ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        temporary = Path(directory) / "submission.csv"
        frame.to_csv(temporary, index=False)
        output.hardlink_to(temporary)

"""Read the competition inputs using the original training cutoff."""

from pathlib import Path

import pandas as pd

from src.evaluation import validate_id_order

TRAIN_START = "2017-01-01"
FORECAST_ORIGIN = pd.Timestamp("2017-08-15")
ROLLING_INPUTS = (
    "train.csv",
    "test.csv",
    "stores.csv",
    "items.csv",
    "oil.csv",
    "sample_submission.csv",
)


def load_training_data(path: Path) -> pd.DataFrame:
    """Read in million-row chunks and retain rows from January 2017 onward."""
    chunks = []
    with pd.read_csv(path, parse_dates=["date"], chunksize=1_000_000) as reader:
        for chunk in reader:
            filtered = chunk.loc[chunk["date"] >= TRAIN_START]
            if not filtered.empty:
                chunks.append(filtered)

    if not chunks:
        raise ValueError(f"{path} contains no training rows from {TRAIN_START} onward")

    training = pd.concat(chunks, ignore_index=True)
    if training["date"].max() > FORECAST_ORIGIN:
        raise ValueError(f"{path} contains training rows after the forecast origin")
    return training


def load_test_data(path: Path) -> pd.DataFrame:
    """Read Kaggle's forecast inputs and retain their original row order."""
    test = pd.read_csv(path, parse_dates=["date"])
    if test.empty or not (test["date"] > FORECAST_ORIGIN).all():
        raise ValueError(f"{path} must contain test rows after the forecast origin")
    return test


def load_rolling_test_inputs(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read dated test rows and the template after checking all rolling inputs."""
    for name in ROLLING_INPUTS:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing rolling LightGBM input {path}")
    test = load_test_data(directory / "test.csv")
    sample = pd.read_csv(directory / "sample_submission.csv")
    validate_id_order(test["id"], sample["id"])
    return test, sample


def load_rolling_metadata(
    directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the store, item and oil tables used by the historical experiment."""
    stores = pd.read_csv(directory / "stores.csv")
    items = pd.read_csv(directory / "items.csv")
    oil = pd.read_csv(directory / "oil.csv", parse_dates=["date"])
    return stores, items, oil

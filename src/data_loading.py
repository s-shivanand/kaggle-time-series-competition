"""Read the competition inputs using the original training cutoff."""

from pathlib import Path

import pandas as pd

TRAIN_START = "2017-01-01"
FORECAST_ORIGIN = pd.Timestamp("2017-08-15")


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

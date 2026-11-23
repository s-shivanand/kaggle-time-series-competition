"""Prepare the basic notebook inputs and scikit-learn column transformation."""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.features import (
    ROLLING_LAGS,
    ROLLING_WINDOWS,
    SERIES_KEYS,
    add_calendar_features,
    with_calendar_columns,
)

BasicRun = Literal["basic-lightgbm", "basic-xgboost"]
BASIC_RUNS = ("basic-lightgbm", "basic-xgboost")
NUMERIC_FEATURES = ("onpromotion", "year", "month", "day")
CATEGORICAL_FEATURES = ("store_nbr", "item_nbr")
ROLLING_NUMERIC_FEATURES = (
    *NUMERIC_FEATURES,
    "class",
    "cluster",
    "perishable",
    "dcoilwtico",
    *(f"lag{lag}" for lag in ROLLING_LAGS),
)
ROLLING_CATEGORICAL_FEATURES = (*CATEGORICAL_FEATURES, "city", "state", "type", "family")
ROLLING_DTYPES = {
    "year": "int16",
    "month": "int8",
    "day": "int8",
    "class": "int16",
    "perishable": "int8",
    "cluster": "int8",
    "dcoilwtico": "float32",
}


def prepare_training_data(
    frame: pd.DataFrame, run: BasicRun
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the original training casts and the model-specific target clipping."""
    frame = frame.drop(columns="id").astype(
        {"store_nbr": "int8", "item_nbr": "int32", "unit_sales": "float32"}
    )
    frame["onpromotion"] = _map_promotions(frame["onpromotion"])
    # The notebooks cast the training year to int8, so 2017 becomes -31.
    frame = add_calendar_features(frame).astype(
        {"year": "int8", "month": "int8", "day": "int8"}
    )
    target = frame["unit_sales"]
    if run == "basic-xgboost":
        target = np.clip(target, 0, None)
    return frame.drop(columns="unit_sales"), target


def prepare_test_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare test features with the notebook test-side calendar dtypes."""
    frame = add_calendar_features(frame.drop(columns="id"))
    frame["onpromotion"] = _map_promotions(frame["onpromotion"])
    return frame.astype({"store_nbr": "int8", "item_nbr": "int32"})


def build_preprocessor() -> ColumnTransformer:
    """Build the original numerical and categorical transformations in order."""
    return _build_preprocessor(
        SimpleImputer(strategy="mean"), NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )


def build_rolling_preprocessor(
    windows: tuple[int, ...] = ROLLING_WINDOWS,
) -> ColumnTransformer:
    """Build the historical rolling column transformations in feature order."""
    numeric_features = (*ROLLING_NUMERIC_FEATURES, *(f"roll_mean_{n}" for n in windows))
    return _build_preprocessor(
        SimpleImputer(strategy="constant", fill_value=0),
        numeric_features,
        ROLLING_CATEGORICAL_FEATURES,
    )


def _build_preprocessor(
    numeric: SimpleImputer,
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
) -> ColumnTransformer:
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, list(numeric_features)),
            ("cat", categorical, list(categorical_features)),
        ]
    )


def _map_promotions(flags: pd.Series) -> pd.Series:
    return flags.map({False: 0, True: 1}, na_action="ignore").astype("int8")


def clean_rolling_training_sales(frame: pd.DataFrame) -> pd.DataFrame:
    """Clip and log-transform sales, cast columns, and sort the training series."""
    frame = frame.drop(columns="id")
    frame["unit_sales"] = np.log1p(np.maximum(0, frame["unit_sales"]))
    frame = frame.astype(
        {"store_nbr": "int8", "item_nbr": "int32", "unit_sales": "float32"}
    )
    frame["onpromotion"] = _map_promotions(frame["onpromotion"])
    return frame.sort_values([*SERIES_KEYS, "date"]).reset_index(drop=True)


def _join_rolling_metadata(
    frame: pd.DataFrame, stores: pd.DataFrame, items: pd.DataFrame, oil: pd.DataFrame
) -> pd.DataFrame:
    return (
        frame.merge(stores, on="store_nbr", how="left")
        .merge(items, on="item_nbr", how="left")
        .merge(oil, on="date", how="left")
    )


def prepare_rolling_training_data(
    frame: pd.DataFrame,
    stores: pd.DataFrame,
    items: pd.DataFrame,
    oil: pd.DataFrame,
    *,
    windows: tuple[int, ...] = ROLLING_WINDOWS,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Join the original training inputs and return features, target, and weights."""
    frame = _join_rolling_metadata(frame, stores, items, oil)
    weights = 1 + 0.25 * frame["perishable"]
    frame = add_calendar_features(frame).astype(_rolling_dtypes(windows))
    return frame.drop(columns="unit_sales"), frame["unit_sales"], weights


def prepare_rolling_test_data(
    frame: pd.DataFrame,
    stores: pd.DataFrame,
    items: pd.DataFrame,
    oil: pd.DataFrame,
    *,
    windows: tuple[int, ...] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Apply the original test-side joins, calendar features, and casts."""
    # Keep the date for the oil join, and append calendar columns before metadata.
    frame = _join_rolling_metadata(with_calendar_columns(frame), stores, items, oil)
    frame["onpromotion"] = frame["onpromotion"].map({False: 0, True: 1}, na_action="ignore")
    return frame.astype(
        {
            "store_nbr": "int8",
            "item_nbr": "int32",
            "onpromotion": "float32",
            **_rolling_dtypes(windows),
        }
    ).drop(columns="date")


def _rolling_dtypes(windows: tuple[int, ...]) -> dict[str, str]:
    return {**ROLLING_DTYPES, **{f"roll_mean_{n}": "float32" for n in windows}}

"""Prepare the basic notebook inputs and scikit-learn column transformation."""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.features import add_calendar_features

BasicRun = Literal["basic-lightgbm", "basic-xgboost"]
BASIC_RUNS = ("basic-lightgbm", "basic-xgboost")
NUMERIC_FEATURES = ("onpromotion", "year", "month", "day")
CATEGORICAL_FEATURES = ("store_nbr", "item_nbr")


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
            ("num", SimpleImputer(strategy="mean"), list(NUMERIC_FEATURES)),
            ("cat", categorical, list(CATEGORICAL_FEATURES)),
        ]
    )


def _map_promotions(flags: pd.Series) -> pd.Series:
    return flags.map({False: 0, True: 1}, na_action="ignore").astype("int8")

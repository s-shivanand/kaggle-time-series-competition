"""Fit the original basic models and historical rolling models."""

from typing import TYPE_CHECKING

import pandas as pd
from sklearn.pipeline import Pipeline

from src.features import XGBOOST_ROLLING_WINDOWS
from src.preprocessing import (
    BasicRun,
    build_preprocessor,
    build_rolling_preprocessor,
    prepare_training_data,
)

if TYPE_CHECKING:
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor


def _build_lightgbm() -> LGBMRegressor:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=1000, verbose=10, random_state=42, learning_rate=0.03
    )


def _build_xgboost(learning_rate: float) -> XGBRegressor:
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",
        n_jobs=-1,
        random_state=42,
    )


def build_pipeline(run: BasicRun) -> Pipeline:
    """Build a notebook pipeline with its original explicit model parameters."""
    if run == "basic-lightgbm":
        model = _build_lightgbm()
    elif run == "basic-xgboost":
        model = _build_xgboost(learning_rate=0.3)
    else:
        raise ValueError(f"Unknown notebook run {run!r}")

    return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])


def fit_basic_model(frame: pd.DataFrame, run: BasicRun) -> Pipeline:
    """Fit on all retained training rows, as in the source notebooks."""
    features, target = prepare_training_data(frame, run)
    return build_pipeline(run).fit(features, target)


def fit_rolling_lightgbm(
    features: pd.DataFrame, target: pd.Series, weights: pd.Series
) -> Pipeline:
    """Fit the historical rolling LightGBM pipeline on all retained rows."""
    pipeline = Pipeline(
        [("preprocessor", build_rolling_preprocessor()), ("model", _build_lightgbm())]
    )
    return pipeline.fit(features, target, model__sample_weight=weights)


def fit_rolling_xgboost(
    features: pd.DataFrame, target: pd.Series, weights: pd.Series
) -> Pipeline:
    """Fit the weighted XGBoost configuration from the original notebook."""
    pipeline = Pipeline(
        [
            ("preprocessor", build_rolling_preprocessor(XGBOOST_ROLLING_WINDOWS)),
            ("model", _build_xgboost(learning_rate=0.03)),
        ]
    )
    return pipeline.fit(features, target, model__sample_weight=weights)

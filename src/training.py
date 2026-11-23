"""Fit the original basic LightGBM and XGBoost configurations."""

import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing import BasicRun, build_preprocessor, prepare_training_data


def build_pipeline(run: BasicRun) -> Pipeline:
    """Build a notebook pipeline with its original explicit model parameters."""
    if run == "basic-lightgbm":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            n_estimators=1000, verbose=10, random_state=42, learning_rate=0.03
        )
    elif run == "basic-xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.3,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown notebook run {run!r}")

    return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])


def fit_basic_model(frame: pd.DataFrame, run: BasicRun) -> Pipeline:
    """Fit on all retained training rows, as in the source notebooks."""
    features, target = prepare_training_data(frame, run)
    return build_pipeline(run).fit(features, target)

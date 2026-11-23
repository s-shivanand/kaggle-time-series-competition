"""Calendar and rolling features for the original forecasting workflows."""

import pandas as pd

ROLLING_LAGS = (3, 7, 14)
XGBOOST_ROLLING_WINDOWS = (3, 7, 14)
ROLLING_WINDOWS = (*XGBOOST_ROLLING_WINDOWS, 21)
SERIES_KEYS = ["store_nbr", "item_nbr"]


def add_rolling_sales_features(
    frame: pd.DataFrame, *, windows: tuple[int, ...] = ROLLING_WINDOWS
) -> None:
    """Add the historical observation-based lags and same-row means in place."""
    grouped_sales = frame.groupby(SERIES_KEYS)["unit_sales"]
    for lag in ROLLING_LAGS:
        frame[f"lag{lag}"] = grouped_sales.transform(lambda sales: sales.shift(lag))
    for window in windows:
        frame[f"roll_mean_{window}"] = grouped_sales.transform(
            lambda sales: sales.rolling(window, min_periods=1).mean()
        )


def build_rolling_test_history(
    training: pd.DataFrame,
    test: pd.DataFrame,
    *,
    windows: tuple[int, ...] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Recompute rolling features over the last 60 days and unlabeled test rows."""
    last_train_date = training["date"].max()
    cutoff = last_train_date - pd.Timedelta(days=60)
    recent = training[training["date"] >= cutoff].copy()
    combined = pd.concat([recent, test.drop(columns="id")], ignore_index=True)
    del recent
    combined = combined.sort_values([*SERIES_KEYS, "date"]).reset_index(drop=True)
    add_rolling_sales_features(combined, windows=windows)
    return combined[combined["date"] > last_train_date].copy()


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add year, month and day, then drop the source date column."""
    return with_calendar_columns(frame).drop(columns="date")


def with_calendar_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Append year, month and day while retaining the date for metadata joins."""
    dates = frame["date"].dt
    return frame.assign(year=dates.year, month=dates.month, day=dates.day)

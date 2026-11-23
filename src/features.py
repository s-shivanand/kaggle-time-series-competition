"""Calendar features used by both basic notebooks."""

import pandas as pd


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add year, month and day, then drop the source date column."""
    dates = frame["date"].dt
    return frame.assign(year=dates.year, month=dates.month, day=dates.day).drop(
        columns="date"
    )

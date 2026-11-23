"""Extract the original EDA example and its saved basic LightGBM forecast."""

import argparse
from hashlib import file_digest
from pathlib import Path

import pandas as pd

from src.evaluation import validate_submission

STORE = 1
ITEM = 211203
EDA_START = "2016-08-15"
LIGHTGBM_SHA256 = "2cda2e23eb3444517c307ccd1ea593ca1621ce758dede4adb32516690afbc880"


def prepare_example(data_dir: Path, submission_path: Path, output_dir: Path) -> None:
    """Export a small date-indexed history and forecast without fitting a model."""
    with submission_path.open("rb") as source:
        if file_digest(source, "sha256").hexdigest() != LIGHTGBM_SHA256:
            raise ValueError(f"{submission_path} differs from the audited LightGBM CSV")

    test = pd.read_csv(
        data_dir / "test.csv", usecols=["id", "date", "store_nbr", "item_nbr"]
    )
    submission = pd.read_csv(submission_path, float_precision="round_trip")
    validate_submission(submission, test["id"])
    selected = test.loc[(test["store_nbr"] == STORE) & (test["item_nbr"] == ITEM)]
    forecast = selected[["id", "date"]].merge(
        submission, on="id", validate="one_to_one"
    )
    forecast = forecast.sort_values("date")
    expected_dates = pd.date_range("2017-08-16", "2017-08-31").strftime("%Y-%m-%d")
    if forecast["date"].tolist() != expected_dates.tolist():
        raise ValueError("The saved LightGBM example must cover all 16 forecast dates")

    parts = []
    with pd.read_csv(
        data_dir / "train.csv",
        usecols=["date", "store_nbr", "item_nbr", "unit_sales"],
        chunksize=500_000,
    ) as reader:
        for chunk in reader:
            mask = (
                (chunk["date"] >= EDA_START)
                & (chunk["store_nbr"] == STORE)
                & (chunk["item_nbr"] == ITEM)
            )
            selected = chunk.loc[mask, ["date", "unit_sales"]]
            if not selected.empty:
                parts.append(selected)

    if not parts:
        raise ValueError("The training CSV does not contain the notebook's EDA example")
    history = pd.concat(parts, ignore_index=True).sort_values("date")
    if len(history) != 362 or history["date"].duplicated().any():
        raise ValueError("Expected 362 distinct historical observations")
    if history["date"].iloc[-1] != "2017-08-15" or history["date"].iloc[0] != EDA_START:
        raise ValueError("Unexpected date range for the historical example")

    output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(output_dir / "example_history.csv", index=False)
    forecast.to_csv(output_dir / "example_forecast.csv", index=False)


def main() -> None:
    """Read local artifact paths from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    try:
        prepare_example(args.data_dir, args.submission, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

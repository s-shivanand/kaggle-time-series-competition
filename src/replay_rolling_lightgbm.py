"""Reproduce the archived rolling LightGBM experiment with Python modules."""

import argparse
import logging
from pathlib import Path

from src.data_loading import (
    load_rolling_metadata,
    load_rolling_test_inputs,
    load_training_data,
)
from src.evaluation import compare_submission_reference, validate_reference_ids
from src.features import add_rolling_sales_features, build_rolling_test_history
from src.prediction import predict_rolling_submission, write_submission
from src.preprocessing import (
    clean_rolling_training_sales,
    prepare_rolling_test_data,
    prepare_rolling_training_data,
)
from src.training import fit_rolling_lightgbm

LOGGER = logging.getLogger(__name__)


def replay(data_dir: Path, output: Path, reference: Path | None = None) -> None:
    """Export the historical result to a new file and optionally compare it."""
    if output.exists():
        raise FileExistsError(f"Choose a new output path. {output} already exists")

    test, sample = load_rolling_test_inputs(data_dir)
    if reference is not None:
        validate_reference_ids(reference, sample["id"])
    stores, items, oil = load_rolling_metadata(data_dir)

    LOGGER.warning(
        "Historical reproduction preserves same-row rolling averages, "
        "forecast-date oil inputs, and positional assignment to the sample IDs."
    )
    LOGGER.info("Loading and cleaning training sales in %s", data_dir)
    training = clean_rolling_training_sales(load_training_data(data_dir / "train.csv"))

    LOGGER.info("Building rolling features for training and test")
    add_rolling_sales_features(training)
    test_history = build_rolling_test_history(training, test)
    del test

    LOGGER.info("Joining training metadata and preparing model inputs")
    features, target, weights = prepare_rolling_training_data(training, stores, items, oil)
    del training

    LOGGER.info("Fitting rolling LightGBM on %s rows", f"{len(target):,}")
    pipeline = fit_rolling_lightgbm(features, target, weights)
    del features, target, weights

    LOGGER.info("Predicting %s test rows", f"{len(test_history):,}")
    test_features = prepare_rolling_test_data(test_history, stores, items, oil)
    del test_history
    result = predict_rolling_submission(pipeline, test_features, sample)
    write_submission(result, sample["id"], output)
    LOGGER.info("Wrote %s", output)
    if reference is not None:
        LOGGER.info("%s", compare_submission_reference(output, reference))


def main() -> None:
    """Run the modular historical workflow from the repository root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path, help="Saved submission to compare")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        replay(args.data_dir, args.output, args.reference)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

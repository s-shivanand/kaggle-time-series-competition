"""Replay either basic notebook from a repository-root command."""

import argparse
import logging
from pathlib import Path
from typing import cast

import pandas as pd

from src.data_loading import load_test_data, load_training_data
from src.evaluation import validate_id_order
from src.prediction import predict_submission, write_submission
from src.preprocessing import BASIC_RUNS, BasicRun
from src.training import fit_basic_model

LOGGER = logging.getLogger(__name__)


def run_notebook(run: BasicRun, data_dir: Path, output: Path) -> None:
    """Load, fit, and export a single basic notebook run."""
    if output.exists():
        raise FileExistsError(f"Choose a new output path. {output} already exists")

    LOGGER.info("Checking forecast inputs in %s", data_dir)
    test = load_test_data(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    validate_id_order(test["id"], sample["id"])

    LOGGER.info("Reading training history from %s", data_dir / "train.csv")
    training = load_training_data(data_dir / "train.csv")
    LOGGER.info("Fitting %s on %s rows", run, f"{len(training):,}")
    pipeline = fit_basic_model(training, run)
    del training

    LOGGER.info("Predicting %s test rows", f"{len(test):,}")
    submission = predict_submission(pipeline, test, sample)
    write_submission(submission, sample["id"], output)
    LOGGER.info("Wrote %s", output)


def main() -> None:
    """Parse the replay arguments and report input or output errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, choices=BASIC_RUNS)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run_notebook(cast(BasicRun, args.run), args.data_dir, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

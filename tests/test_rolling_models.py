"""Check the modular rolling workflow against the original notebook."""

import json
import sys
import unittest
from contextlib import chdir, redirect_stdout
from io import StringIO
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import joblib
import pandas as pd

from src.prediction import predict_rolling_submission
from src.replay_rolling_lightgbm import replay
from src.training import save_trained_pipeline

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_NOTEBOOK = ROOT / "notebooks" / "SKS_xgboost_moving_avg_minimal.ipynb"
REPRODUCTION_NOTEBOOK = ROOT / "notebooks" / "01_lightgbm.ipynb"
SUBMISSION_FILENAME = "lightgbm_mean_lag3_7_14_21_with_features.csv"
XGBOOST_NOTEBOOK = ROOT / "notebooks" / "02_xgboost.ipynb"
XGBOOST_SUBMISSION = "xgboost_mean_lag3_7_14_with_features.csv"


def prepare_fixture(directory: Path) -> None:
    """Create a dated two-series fixture with a different order after sorting."""
    directory.mkdir(parents=True)
    training_dates = pd.date_range("2017-06-21", "2017-08-15")
    test_dates = pd.date_range("2017-08-16", "2017-08-18")
    training = pd.DataFrame(
        [
            (date, store, 101, (day + store) % 9 - 1, day % 2 == 0)
            for day, date in enumerate(training_dates)
            for store in (1, 2)
            if day % 5 != 0 or store == 1
        ],
        columns=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
    )
    training.insert(0, "id", range(len(training)))
    training.to_csv(directory / "train.csv", index=False)

    test = pd.DataFrame(
        [(date, store, 101, store == 1) for date in test_dates for store in (1, 2)],
        columns=["date", "store_nbr", "item_nbr", "onpromotion"],
    )
    test.insert(0, "id", range(len(training), len(training) + len(test)))
    test.to_csv(directory / "test.csv", index=False)
    test[["id"]].assign(unit_sales=0).to_csv(
        directory / "sample_submission.csv", index=False
    )
    pd.DataFrame(
        [(1, "Quito", "Pichincha", "A", 1), (2, "Guayaquil", "Guayas", "B", 2)],
        columns=["store_nbr", "city", "state", "type", "cluster"],
    ).to_csv(directory / "stores.csv", index=False)
    pd.DataFrame([(101, "Grocery", 1001, 1)], columns=[
        "item_nbr", "family", "class", "perishable"
    ]).to_csv(directory / "items.csv", index=False)
    oil = pd.DataFrame(
        {"date": training_dates.append(test_dates), "dcoilwtico": range(50, 109)}
    )
    oil.loc[::3, "dcoilwtico"] = float("nan")
    oil.to_csv(directory / "oil.csv", index=False)


def execute_notebook_cells(
    notebook: Path, directory: Path, *, activate_xgboost: bool = False
) -> dict[str, Any]:
    """Execute a notebook's code cells on the fixture without a Jupyter dependency."""
    cells = json.loads(notebook.read_bytes())["cells"]
    namespace: dict[str, object] = {}
    with chdir(directory), patch.object(sys, "path", sys.path.copy()):
        for number, cell in enumerate(cells, start=1):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if activate_xgboost:
                source = xgboost_source_cell(number, source)
            exec(source, namespace)
    return namespace


def xgboost_source_cell(number: int, source: str) -> str:
    """Apply the requested edits to the original notebook for the parity test."""
    if number in (37, 53, 93):  # Training mean, test mean, and LightGBM model.
        return ""
    if number in (78, 83, 105):  # Training cast, feature selection, and test cast.
        return "\n".join(line for line in source.splitlines() if "roll_mean_21" not in line)
    if number == 94:
        return "\n".join(
            line[1:].removeprefix(" ") if line.startswith("#") else line
            for line in source.splitlines()
        )
    if number == 116:
        return source.replace("lightgbm_lag3_7_14_21_with_features.csv", XGBOOST_SUBMISSION)
    return source


class RollingModelTests(unittest.TestCase):
    def assert_saved_pipeline_matches(
        self, model_path: Path, result_path: Path, test_features: pd.DataFrame
    ) -> None:
        pipeline = joblib.load(model_path)
        sample = pd.read_csv(result_path)
        forecast = predict_rolling_submission(pipeline, test_features, sample)
        self.assertEqual(forecast.to_csv(index=False).encode(), result_path.read_bytes())
        model_bytes = model_path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_trained_pipeline(pipeline, model_path)
        self.assertEqual(model_path.read_bytes(), model_bytes)

    def test_cli_runner_and_new_notebook_match_the_original(self) -> None:
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            notebooks = root / "notebooks"
            notebooks.mkdir()
            inputs = root / "data" / "raw" / "favorita-grocery-sales-forecasting"
            prepare_fixture(inputs)

            original = execute_notebook_cells(ORIGINAL_NOTEBOOK, notebooks)
            reference = inputs / "lightgbm_lag3_7_14_21_with_features.csv"
            output = root / "cli" / SUBMISSION_FILENAME
            with self.assertLogs("src.replay_rolling_lightgbm", level="INFO") as logs:
                replay(inputs, output, reference)
            self.assertIn("6 rows, 6 exact", "\n".join(logs.output))
            self.assertEqual(output.read_bytes(), reference.read_bytes())
            self.assertGreater(pd.read_csv(output)["unit_sales"].nunique(), 1)
            with self.assertRaises(FileExistsError):
                replay(inputs, reference)

            (root / "pyproject.toml").touch()
            archived = root / "data" / "submission" / SUBMISSION_FILENAME
            archived.parent.mkdir()
            copyfile(reference, archived)
            published = root / "models" / "lightgbm.joblib"
            published.parent.mkdir()
            published.write_bytes(b"published model")
            with redirect_stdout(StringIO()) as notebook_output:
                execute_notebook_cells(REPRODUCTION_NOTEBOOK, notebooks)
            self.assertIn("6 rows, 6 exact", notebook_output.getvalue())
            reproduced = root / "data" / "reproduction" / SUBMISSION_FILENAME
            self.assertEqual(reproduced.read_bytes(), reference.read_bytes())
            self.assert_saved_pipeline_matches(
                root / "models" / "reproduced" / "lightgbm.joblib",
                reproduced,
                original["X_test"],
            )
            self.assertEqual(published.read_bytes(), b"published model")

    def test_xgboost_notebook_matches_the_edited_source_recipe(self) -> None:
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            notebooks = root / "notebooks"
            notebooks.mkdir()
            (root / "pyproject.toml").touch()
            inputs = root / "data" / "raw" / "favorita-grocery-sales-forecasting"
            prepare_fixture(inputs)

            original = execute_notebook_cells(
                ORIGINAL_NOTEBOOK, notebooks, activate_xgboost=True
            )
            archived = root / "data" / "submission" / XGBOOST_SUBMISSION
            archived.parent.mkdir()
            copyfile(inputs / XGBOOST_SUBMISSION, archived)

            with redirect_stdout(StringIO()) as notebook_output:
                execute_notebook_cells(XGBOOST_NOTEBOOK, notebooks)
            self.assertIn("6 rows, 6 exact", notebook_output.getvalue())
            result = root / "data" / "reproduction" / XGBOOST_SUBMISSION
            self.assertEqual(result.read_bytes(), archived.read_bytes())
            self.assertGreater(pd.read_csv(result)["unit_sales"].nunique(), 1)
            self.assert_saved_pipeline_matches(
                root / "models" / "reproduced" / "xgboost.joblib",
                result,
                original["X_test"],
            )


if __name__ == "__main__":
    unittest.main()

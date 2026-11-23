"""Check the runnable notebook paths and demonstrate archived feature defects."""

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.data_loading import load_test_data, load_training_data
from src.evaluation import validate_submission
from src.prediction import predict_submission, write_submission
from src.preprocessing import (
    build_preprocessor,
    prepare_test_data,
    prepare_training_data,
)
from src.run import run_notebook
from src.training import build_pipeline

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def training_fixture() -> pd.DataFrame:
    """Return a small series with dates on both sides of the original cutoff."""
    return pd.DataFrame(
        {
            "id": [9, 11, 10],
            "date": pd.to_datetime(["2016-12-31", "2017-01-01", "2017-08-15"]),
            "store_nbr": [1, 2, 1],
            "item_nbr": [100001, 200001, 100001],
            "unit_sales": [100.0, -2.0, 3.125],
            "onpromotion": [True, False, True],
        }
    )


def forecast_fixture() -> pd.DataFrame:
    """Return forecast rows in submission order with an unseen store and item."""
    return pd.DataFrame(
        {
            "id": [20, 21],
            "date": pd.to_datetime(["2017-08-16", "2017-08-16"]),
            "store_nbr": [3, 1],
            "item_nbr": [300001, 100001],
            "onpromotion": [True, False],
        }
    )


class ReplayTests(unittest.TestCase):
    def test_cutoff_order_and_forecast_origin(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sales.csv"
            frame = training_fixture()
            frame.to_csv(path, index=False)
            self.assertEqual(load_training_data(path)["id"].tolist(), [11, 10])
            with self.assertRaisesRegex(ValueError, "after the forecast origin"):
                load_test_data(path)
            frame.loc[2, "date"] = pd.Timestamp("2017-08-16")
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "after the forecast origin"):
                load_training_data(path)

    def test_original_features_and_target_dtypes(self) -> None:
        x, raw = prepare_training_data(training_fixture().iloc[1:], "basic-lightgbm")
        _, clipped = prepare_training_data(training_fixture().iloc[1:], "basic-xgboost")
        x_test = prepare_test_data(forecast_fixture())
        columns = ["store_nbr", "item_nbr", "onpromotion", "year", "month", "day"]
        self.assertEqual(x.columns.tolist(), columns)
        self.assertEqual(x_test.columns.tolist(), columns)
        self.assertEqual(
            x.dtypes.astype(str).tolist(),
            ["int8", "int32", "int8", "int8", "int8", "int8"],
        )
        self.assertEqual(
            x_test.dtypes.astype(str).tolist(),
            ["int8", "int32", "int8", "int32", "int32", "int32"],
        )
        self.assertEqual(x["year"].tolist(), [-31, -31])
        self.assertEqual(x_test["year"].tolist(), [2017, 2017])
        self.assertEqual(x["onpromotion"].tolist(), [0, 1])
        self.assertEqual(x_test["onpromotion"].tolist(), [1, 0])
        self.assertEqual(str(raw.dtype), "float32")
        self.assertEqual(str(clipped.dtype), "float32")
        self.assertEqual(raw.tolist(), [-2.0, 3.125])
        self.assertEqual(clipped.tolist(), [0.0, 3.125])

    def test_preprocessor_order_imputation_and_unknown_categories(self) -> None:
        x, _ = prepare_training_data(training_fixture().iloc[1:], "basic-lightgbm")
        x = x.astype({"onpromotion": "float64", "store_nbr": "float64"})
        x.loc[x.index[0], "onpromotion"] = np.nan
        x.loc[x.index[0], "store_nbr"] = np.nan
        processor = build_preprocessor().fit(x)
        self.assertEqual(processor.named_transformers_["num"].statistics_[0], 1.0)
        categorical = processor.named_transformers_["cat"]
        self.assertEqual(categorical.named_steps["imputer"].statistics_[0], 1.0)
        np.testing.assert_array_equal(
            processor.transform(prepare_test_data(forecast_fixture()))[0],
            [1, 2017, 8, 16, -1, -1],
        )

    def test_parameters_are_the_explicit_notebook_values(self) -> None:
        lightgbm = build_pipeline("basic-lightgbm").named_steps["model"].get_params()
        xgboost = build_pipeline("basic-xgboost").named_steps["model"].get_params()
        for parameters in (lightgbm, xgboost):
            self.assertEqual(parameters["n_estimators"], 1000)
            self.assertEqual(parameters["random_state"], 42)
        self.assertEqual(lightgbm["learning_rate"], 0.03)
        self.assertEqual(lightgbm["verbose"], 10)
        expected = {
            "learning_rate": 0.3,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_jobs": -1,
        }
        self.assertEqual({name: xgboost[name] for name in expected}, expected)

    def test_prediction_clipping_and_reordered_ids(self) -> None:
        pipeline = Mock()
        pipeline.predict.return_value = np.array([-2.0, 4.0])
        test = forecast_fixture()
        sample = pd.DataFrame({"id": [20, 21], "unit_sales": [0, 0]})
        result = predict_submission(pipeline, test, sample)
        self.assertEqual(
            result.to_dict("list"), {"id": [20, 21], "unit_sales": [0.0, 4.0]}
        )
        pipeline.reset_mock()
        with self.assertRaisesRegex(ValueError, "row order"):
            predict_submission(pipeline, test.iloc[::-1], sample)
        pipeline.predict.assert_not_called()

        with TemporaryDirectory() as directory:
            output = Path(directory) / "submission.csv"
            with self.assertRaisesRegex(ValueError, "row order"):
                write_submission(result.iloc[::-1], sample["id"], output)
            self.assertFalse(output.exists())
            for values in ([np.inf, 1], [np.nan, 1], [-1, 1], ["0", "1"]):
                invalid = result.assign(unit_sales=values)
                with (
                    self.subTest(values=values),
                    self.assertRaisesRegex(ValueError, "unit_sales"),
                ):
                    write_submission(invalid, sample["id"], output)
            self.assertFalse(output.exists())
            write_submission(result, sample["id"], output)
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                write_submission(result.assign(unit_sales=1), sample["id"], output)
            self.assertEqual(output.read_bytes(), original)

    def test_both_basic_runs_fit_and_export_on_fixture(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            test = forecast_fixture()
            training_fixture().to_csv(root / "train.csv", index=False)
            test.to_csv(root / "test.csv", index=False)
            sample = pd.DataFrame({"id": test["id"], "unit_sales": 0})
            sample.to_csv(root / "sample_submission.csv", index=False)
            for run in ("basic-lightgbm", "basic-xgboost"):
                with self.subTest(run=run), redirect_stdout(StringIO()):
                    output = root / "output" / f"{run}.csv"
                    run_notebook(run, root, output)
                    validate_submission(pd.read_csv(output), sample["id"])


def run_archived_cell(
    notebook: str, start: str, namespace: dict[str, object]
) -> None:
    """Run a named feature cell from a preserved notebook on synthetic rows."""
    cells = json.loads((NOTEBOOKS / notebook).read_text(encoding="utf-8"))["cells"]
    sources = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
    (source,) = (source for source in sources if source.startswith(start))
    exec(compile(source, notebook, "exec"), namespace)


class HistoricalRiskTests(unittest.TestCase):
    def test_lag_16_counts_observations_across_missing_dates(self) -> None:
        train = pd.DataFrame(
            {
                "store_nbr": 1,
                "item_nbr": 2,
                "date": pd.date_range("2017-01-01", periods=17, freq="2D"),
                "unit_sales": np.arange(1.0, 18.0),
            }
        )
        grouped = train.groupby(["store_nbr", "item_nbr"])["unit_sales"]
        run_archived_cell(
            "SKS_xgboost_moving_avg.ipynb",
            'df_train["lag16"] =',
            {"df_train": train, "grouped_data": grouped},
        )
        self.assertEqual(train["lag16"].iloc[-1], 1.0)
        self.assertEqual((train["date"].iloc[-1] - train["date"].iloc[0]).days, 32)

    def test_latest_training_window_includes_same_row_target(self) -> None:
        notebook = "SKS_xgboost_moving_avg_minimal.ipynb"
        example = pd.DataFrame(
            {"store_nbr": [1, 1], "item_nbr": [2, 2], "unit_sales": [1.0, 9.0]}
        )
        grouped = example.groupby(["store_nbr", "item_nbr"])["unit_sales"]
        run_archived_cell(
            notebook,
            'df_train["roll_mean_3"] =',
            {"df_train": example, "grouped_data": grouped},
        )
        self.assertEqual(example["roll_mean_3"].tolist(), [1.0, 5.0])

    def test_both_archived_forecast_sorts_change_position(self) -> None:
        test = forecast_fixture()
        for notebook in (
            "SKS_xgboost_moving_avg.ipynb",
            "SKS_xgboost_moving_avg_minimal.ipynb",
        ):
            namespace = {"pd": pd, "df_recent_train": test.iloc[:0], "df_test": test}
            with self.subTest(notebook=notebook):
                run_archived_cell(notebook, "# concatenate vertically", namespace)
                self.assertEqual(namespace["full_df"]["id"].tolist(), [21, 20])


if __name__ == "__main__":
    unittest.main()

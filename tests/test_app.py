"""Smoke-test the app in an isolated directory without the competition data."""

import unittest
from contextlib import chdir
from pathlib import Path
from shutil import copy, copytree
from tempfile import TemporaryDirectory

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    def test_app_without_raw_data_and_with_or_without_forecast_excerpt(self) -> None:
        with TemporaryDirectory() as directory, chdir(directory):
            isolated = Path(directory)
            copy(ROOT / "app.py", isolated)
            copytree(ROOT / "artifacts", isolated / "artifacts")
            (isolated / "src").mkdir()
            copy(ROOT / "src" / "evaluation.py", isolated / "src")

            app = AppTest.from_file(str(isolated / "app.py")).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertFalse(app.info)
            self.assertEqual(
                [tab.label for tab in app.tabs],
                [
                    "Data cleaning", "Data exploration", "Feature engineering",
                    "Model development", "Evaluation", "Deployment",
                ],
            )
            cleaning, exploration, features, models, evaluation, deployment = app.tabs
            self.assertEqual(len(cleaning.get("vega_lite_chart")), 1)
            self.assertEqual(len(exploration.get("vega_lite_chart")), 1)
            self.assertEqual(len(features.table), 1)
            self.assertEqual(len(models.table), 1)
            self.assertEqual(len(evaluation.get("vega_lite_chart")), 1)
            self.assertEqual(len(deployment.get("vega_lite_chart")), 1)
            scores = app.dataframe[0].value
            self.assertEqual(
                scores["private_score"].tolist(),
                [1.48557, 1.47656, 1.19868, 1.09747, 1.11764],
            )
            self.assertEqual(
                scores["public_score"].tolist(),
                [1.45230, 1.44706, 1.17484, 1.11674, 1.13637],
            )

            history = pd.read_csv(isolated / "artifacts" / "example_history.csv")
            self.assertEqual(len(history), 362)
            self.assertAlmostEqual(history["unit_sales"].autocorr(lag=1), 0.102156, places=6)
            self.assertAlmostEqual(history["unit_sales"].autocorr(lag=7), 0.118836, places=6)

            forecast = isolated / "artifacts" / "example_forecast.csv"
            excerpt = pd.read_csv(forecast, parse_dates=["date"])
            expected_dates = pd.date_range("2017-08-16", "2017-08-31")
            self.assertEqual(excerpt["date"].tolist(), expected_dates.tolist())
            self.assertEqual(excerpt["id"].iloc[0], 125497188)
            self.assertEqual(excerpt["id"].iloc[-1], 128656998)
            self.assertAlmostEqual(excerpt["unit_sales"].iloc[0], 3.144170, places=6)

            forecast.unlink()
            app.run(timeout=30)
            self.assertFalse(app.exception)
            self.assertIn("example_forecast.csv", app.info[0].value)
            self.assertEqual(len(app.tabs[-1].get("vega_lite_chart")), 0)


if __name__ == "__main__":
    unittest.main()

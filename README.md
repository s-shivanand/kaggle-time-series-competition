# Favorita Grocery Sales Forecasting

This project forecasts daily unit sales for store–item pairs across 54 Ecuadorian stores. The original Kaggle forecasts cover August 16–31, 2017. The research used LightGBM and XGBoost, starting with calendar and promotion features and exploring richer store, item, oil, and historical-sales features. [Kaggle competition](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting)

The lowest reported private score across five late submissions was **1.09747** for an XGBoost submission. The [baseline audit](docs/BASELINE.md) records both leaderboard splits, available recipes, the absence of local validation, and the limitations found in the rolling experiments. The repo provides two basic workflows and two historical rolling reproduction notebooks.

## Browse the results

[Open the live Streamlit dashboard](https://gx3zmjtv3cwmufmnpwhjbe.streamlit.app/).

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run commands from the repository root. The project pins Python 3.14.4 in `.python-version` and uses `uv.lock` for reproducible installs. `uv sync --locked` sets up the default environment for modeling, the app, and tests.

```bash
uv run --locked --no-group models streamlit run app.py
```

The app walks through data cleaning, exploration, feature engineering, model development, evaluation, and deployment. Each stage includes a visual based on the original research. The app reads three small CSVs from `artifacts/` and needs neither the competition download nor a fitted model. The deployment tab shows how the local app serves saved results and displays the 16-day basic LightGBM forecast for the same store–item pair used in the historical exploration.

## Reproduce a basic notebook run

Place the extracted competition files in `data/raw/favorita-grocery-sales-forecasting/`. Keep the original submission CSVs separately in `data/submission/`. The local `data/` directory is Git-ignored. [Kaggle data files](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data)

```bash
uv run --locked --no-group app python -m src.run --run basic-lightgbm \
  --data-dir data/raw/favorita-grocery-sales-forecasting \
  --output data/submission/reproduced_lightgbm.csv
```

Use `--run basic-xgboost` and choose a new output path for the other basic notebook. On macOS the model libraries may need the OpenMP runtime. The full run reads the 4.7 GB training CSV and fits on 23.8 million rows. It needs substantial memory and time. Both pipelines were verified on small fixtures. Full-data parity remains unverified pending approval for an expensive run.

To inspect the notebooks, launch `uv run --locked --group notebooks jupyter lab notebooks/`. The modeling notebooks use paths relative to `notebooks/`. The [baseline audit](docs/BASELINE.md) documents the historical EDA notebook and its machine-specific path.

## Replay the rolling LightGBM submission

The original `lightgbm_mean_lag3_7_14_21_with_features.csv` received **1.11764** on Kaggle’s private leaderboard. The [LightGBM reproduction notebook](notebooks/01_lightgbm.ipynb) calls the Python modules in `src/` to load and clean data, build features, fit the model, compare with the archived submission, and export the new CSV. The full fit uses 23.8 million rows.

From the repository root, launch the notebook in the locked environment. Select its Python kernel and run all cells. The path cell lets you change the data directory, output path, and optional saved reference.

```bash
uv run --locked --no-group app --group notebooks jupyter lab notebooks/01_lightgbm.ipynb
```

The command-line entry point uses the same Python modules.

```bash
uv run --locked --exact --no-group app python -m src.replay_rolling_lightgbm \
  --data-dir data/raw/favorita-grocery-sales-forecasting \
  --output data/reproduction/lightgbm_mean_lag3_7_14_21_with_features.csv \
  --reference data/submission/lightgbm_mean_lag3_7_14_21_with_features.csv
```

Use the optional `--reference` argument when you have the original CSV. The workflow checks template IDs and reports exact matches, matches within `rtol=1e-6` and `atol=1e-6`, and the mean and maximum absolute prediction differences. Choose a fresh output path for each run. Both entry points matched the original notebook byte for byte on a test fixture. Full-data parity remains unverified, and Kaggle’s private labels are unavailable locally. The historical workflow retains same-row rolling averages, forecast-date oil values, and positional assignment to submission IDs. These limitations are documented in the [baseline audit](docs/BASELINE.md#limitations-and-environment).

## Replay the rolling XGBoost submission

The archived `xgboost_mean_lag3_7_14_with_features.csv` received **1.09747** on Kaggle’s private leaderboard. The [XGBoost reproduction notebook](notebooks/02_xgboost.ipynb) uses the XGBoost configuration from `SKS_xgboost_moving_avg_minimal.ipynb` and retains lags and rolling means of 3, 7, and 14 observations.

```bash
uv run --locked --no-group app --group notebooks jupyter lab notebooks/02_xgboost.ipynb
```

Run all cells to write the CSV under `data/reproduction/`. The notebook compares it with the same filename under `data/submission/` when available. It matched the original notebook with the requested XGBoost edits byte for byte on a fixture. The full-data comparison remains unverified.

## Verify the project

Run the same suite as GitHub Actions. The synthetic fixtures exercise both models without competition data.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 uv run --locked python -m unittest discover -s tests -v
```

To regenerate the two chart excerpts from local data and the audited LightGBM CSV, run the command below. It scans the source history without fitting and checks the saved forecast file's SHA-256 before extraction.

```bash
uv run --locked --no-default-groups python -m src.prepare_visualization_data \
  --data-dir data/raw/favorita-grocery-sales-forecasting \
  --submission data/submission/submission_lightgbm.csv \
  --output-dir artifacts
```

# Favorita Grocery Sales Forecasting

This project forecasts daily unit sales for store–item pairs across 54 Ecuadorian stores. The original Kaggle forecasts cover August 16–31, 2017. The research used LightGBM and XGBoost, starting with calendar and promotion features and exploring richer store, item, oil, and historical-sales features. [Kaggle competition](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting)

The lowest reported private score across five late submissions was **1.09747** for an XGBoost submission. The [baseline audit](docs/BASELINE.md) records both leaderboard splits, available recipes, the absence of local validation, and the limitations found in the rolling experiments. The runnable workflows reproduce the two basic notebooks.

## Browse the results

From the repository root, use Python 3.14.4.

```bash
python -m venv .venv-app
source .venv-app/bin/activate
python -m pip install -r requirements-app.txt
python -m streamlit run app.py
```

The app walks through data cleaning, exploration, feature engineering, model development, evaluation, and deployment. Each stage includes a visual based on the original research. The app reads three small CSVs from `artifacts/` and needs neither the competition download nor a fitted model. The deployment tab shows how the local app serves saved results and displays the 16-day basic LightGBM forecast for the same store–item pair used in the historical exploration.

## Reproduce a basic notebook run

Place the extracted competition files in `data/raw/favorita-grocery-sales-forecasting/`. Keep the original submission CSVs separately in `data/submission/`. The local `data/` directory is Git-ignored. [Kaggle data files](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data)

```bash
python -m venv .venv-replay
source .venv-replay/bin/activate
python -m pip install -r requirements.txt
python -m src.run --run basic-lightgbm \
  --data-dir data/raw/favorita-grocery-sales-forecasting \
  --output data/submission/reproduced_lightgbm.csv
```

Use `--run basic-xgboost` and choose a new output path for the other basic notebook. On macOS the model libraries may need the OpenMP runtime. The full run reads the 4.7 GB training CSV and fits on 23.8 million rows. It needs substantial memory and time. Both pipelines were verified on small fixtures. Full-data parity remains unverified pending approval for an expensive run.

For notebook inspection, the local environment used `jupyter==1.1.1`, `ipykernel==7.3.0`, and `matplotlib==3.11.1` in addition to the model dependencies. The modeling notebooks use paths relative to `notebooks/`. The [baseline audit](docs/BASELINE.md) documents the historical EDA notebook and its machine-specific path.

## Verify the project

In either environment, install both requirement files and run the tests. The synthetic fixtures exercise both models without competition data.

```bash
python -m pip install -r requirements.txt -r requirements-app.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
```

To regenerate the two chart excerpts from local data and the audited LightGBM CSV, run the command below. It scans the source history without fitting and checks the saved forecast file's SHA-256 before extraction.

```bash
python -m src.prepare_visualization_data \
  --data-dir data/raw/favorita-grocery-sales-forecasting \
  --submission data/submission/submission_lightgbm.csv \
  --output-dir artifacts
```

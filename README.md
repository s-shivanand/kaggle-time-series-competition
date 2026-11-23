# Favorita Grocery Sales Forecasting

Forecast 16 days of daily sales by item and store for 54 Ecuadorian grocery stores. [Kaggle competition](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting)

**[Open the live Streamlit dashboard](https://kaggle-grocery-sale-forecast.streamlit.app/)**

| Model notebook | Original Kaggle private score ↓ |
| --- | ---: |
| [LightGBM](notebooks/01_lightgbm.ipynb) | 1.11764 |
| [XGBoost](notebooks/02_xgboost.ipynb) | 1.09747 |

## Run locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run this from the repository root. The dashboard uses saved results.

```bash
uv run --locked --no-group models streamlit run app.py
```

## Regenerate submissions

Extract the [Kaggle CSVs](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data) into `data/raw/favorita-grocery-sales-forecasting/`. Open either model notebook and select **Run All**.

```bash
uv run --locked --no-group app --group notebooks jupyter lab notebooks/
```

The notebooks write CSVs to `data/reproduction/`. Both matched their source recipes on small test fixtures. Full-data matches to the original submissions remain unverified.

[Methods, limitations, and command-line runs](docs/BASELINE.md)

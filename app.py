"""Present the original Favorita research and saved competition results."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.evaluation import read_kaggle_scores

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
COMPETITION = "https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting"
STAGES = (
    "Data cleaning",
    "Data exploration",
    "Feature engineering",
    "Model development",
    "Evaluation",
    "Deployment",
)


def show_sales_chart(filename: str, label: str, color: str) -> None:
    """Display a prepared series or explain where the values come from."""
    try:
        series = pd.read_csv(ARTIFACTS / filename, parse_dates=["date"])
    except FileNotFoundError:
        st.info(f"The prepared excerpt `artifacts/{filename}` is unavailable.")
        return

    st.line_chart(
        series.rename(columns={"unit_sales": label}),
        x="date",
        y=label,
        x_label="Date",
        y_label="Units sold",
        color=color,
    )


def show_cleaning() -> None:
    """Show the recorded training cutoff and preprocessing choices."""
    st.header("Data cleaning")
    st.markdown(
        "All five fitted notebooks read training data in million-row chunks and "
        "retain observations from January 1 through August 15, 2017."
    )
    full, retained, duplicates = st.columns(3)
    full.metric("Missing promotion flags · full history", "21,657,651", border=True)
    retained.metric("Missing promotion flags · fitted window", "0", border=True)
    duplicates.metric("Duplicate sales keys · full history", "0", border=True)
    counts = pd.DataFrame(
        {
            "Window": ["Full history", "Fitted window"],
            "Training rows": [125_497_040, 23_808_261],
        }
    )
    st.bar_chart(
        counts,
        x="Window",
        y="Training rows",
        horizontal=True,
        sort=False,
        height=170,
        alt="125,497,040 rows in the full history and 23,808,261 in the fitted window.",
    )
    st.table(
        pd.DataFrame(
            [
                ("Promotion", "Map Boolean flags to 0 or 1."),
                ("Basic LightGBM target", "Keep unit sales and cast to float32."),
                ("Basic XGBoost target", "Clip negative unit sales to zero."),
                ("Enriched targets", "Clip negatives to zero and apply log1p."),
                (
                    "Missing model inputs",
                    "Use the most frequent category. Use numeric means in the "
                    "basic and merge-only runs, and zero in the rolling runs.",
                ),
            ],
            columns=["Input", "Notebook treatment"],
        ),
        hide_index=True,
    )
    st.caption(
        "A sales key is the date, store, and item combination. Counts and "
        "preprocessing choices come from the saved notebooks."
    )


def show_exploration() -> None:
    """Describe the competition inputs and the EDA example."""
    st.header("Daily sales across 54 Ecuadorian stores")
    st.markdown(
        "The full sales history runs from January 2013 to August 2017. All five "
        "fitted notebooks use 23,808,261 observations from January 1 through "
        "August 15, 2017. The basic runs forecast sales from store and item IDs, "
        "promotions, and calendar dates. Enriched experiments also join item "
        "details, store details, and daily oil prices. Holidays and store "
        "transactions were inspected during exploration."
    )
    st.subheader("Observed sales · store 1, item 211203")
    show_sales_chart("example_history.csv", "Observed unit sales", "#167B75")
    st.caption(
        "362 recorded observations from August 2016 to August 2017, drawn from the "
        "historical EDA example. The chart displays recorded dates."
    )
    st.markdown(
        "The example has lag-1 autocorrelation of **0.1022** and lag-7 autocorrelation "
        "of **0.1188**. The lags count observations, so missing dates can extend the "
        "calendar time they span. These values describe the selected series."
    )
    st.link_button("Competition dataset", f"{COMPETITION}/data")


def show_features() -> None:
    """Show how the original feature groups enter the notebook models."""
    st.header("Feature engineering")
    st.markdown(
        "The notebook recipes combine calendar and promotion signals, metadata, "
        "and recorded sales history in different ways. A lag counts rows within "
        "a store–item series, so missing dates can extend its calendar span."
    )
    st.table(
        pd.DataFrame(
            [
                (
                    "Core inputs",
                    "Store and item IDs, promotion, year, month, and day",
                    "All five notebooks",
                ),
                (
                    "Store, item, and oil",
                    "City, state, store type and cluster, item family and class, "
                    "perishability, and dated oil price",
                    "Three enriched notebooks",
                ),
                (
                    "Shifted sales history",
                    "16-row lag and a 7-row mean of the lagged sales",
                    "Rolling XGBoost notebook",
                ),
                (
                    "Lags and rolling means",
                    "Lags of 3, 7, and 14 rows. Unshifted means over "
                    "3, 7, 14, and 21 rows",
                    "Rolling LightGBM notebook",
                ),
            ],
            columns=["Feature group", "What the notebooks computed", "Used by"],
        ),
        hide_index=True,
    )
    st.subheader("How the lag-16 feature works")
    history, lag, average = st.columns(3)
    with history.container(border=True):
        st.markdown("**Observed sales**  \nSort each store–item series by date.")
    with lag.container(border=True):
        st.markdown("**Lag 16**  \nUse sales from 16 observed rows earlier.")
    with average.container(border=True):
        st.markdown("**7-row average**  \nAverage the available values of the lag.")

    st.subheader("Feature timing")
    st.markdown(
        "The rolling LightGBM training means include each row’s own target. The "
        "enriched runs also used actual oil prices dated after the forecast origin. "
        "Both basic notebooks preserve an `int8` cast that wraps the training year "
        "2017 to −31, while the test year stays 2017."
    )


def show_models() -> None:
    """Compare the original model configurations and training choices."""
    st.header("Model development")
    st.markdown(
        "Each notebook fits a scikit-learn preprocessing pipeline and a single "
        "LightGBM or XGBoost regressor on the retained 2017 history. Store and item "
        "identifiers are ordinal-encoded, with unknown categories mapped to −1."
    )
    trees, seed, weight = st.columns(3)
    trees.metric("Trees per run", "1,000", border=True)
    seed.metric("Random seed", "42", border=True)
    weight.metric("Perishable weight · enriched runs", "1.25×", border=True)
    st.table(
        pd.DataFrame(
            [
                ("Basic LightGBM", 6, 0.03, "Raw unit sales"),
                ("Basic XGBoost", 6, 0.30, "Clipped unit sales"),
                ("Enriched XGBoost", 14, 0.30, "Log-transformed and weighted"),
                ("Rolling XGBoost", 16, 0.03, "Log-transformed and weighted"),
                ("Rolling LightGBM", 21, 0.03, "Log-transformed and weighted"),
            ],
            columns=["Notebook run", "Inputs", "Learning rate", "Training target"],
        ),
        hide_index=True,
    )
    st.caption(
        "The two basic notebooks can be replayed from Python. The three enriched "
        "runs remain documented in the research notebooks."
    )


def show_results() -> None:
    """Show the original validation setup and the reported Kaggle scores."""
    st.header("Evaluation")
    training, scoring = st.columns(2)
    with training.container(border=True):
        st.markdown("**Training window**  \nJanuary 1–August 15, 2017")
    with scoring.container(border=True):
        st.markdown("**Kaggle forecast window**  \nAugust 16–31, 2017")
    st.markdown(
        "The notebooks fit on the retained training history through August 15, 2017 "
        "without a local holdout or backtest. The five scores below come from "
        "completed late submissions. Kaggle scores normalized weighted RMSLE, "
        "with a 1.25 weight for perishable items and a 1.00 weight for other "
        "items. Lower scores are better."
    )
    scores = read_kaggle_scores(ARTIFACTS / "kaggle_scores.csv")
    st.subheader("Five reported Kaggle scores")
    short_names = {
        "submission_lightgbm.csv": "Basic LightGBM",
        "submission_xgboost.csv": "XGBoost",
        "xgboost_lag3_7_14_with_features.csv": "XGBoost lag 3/7/14",
        "xgboost_mean_lag3_7_14_with_features.csv": "XGBoost lag and means",
        "lightgbm_mean_lag3_7_14_21_with_features.csv": "LightGBM lag and means",
    }
    chart = scores.assign(Submission=scores["kaggle_submission"].replace(short_names))
    chart = chart.rename(
        columns={"private_score": "Private score", "public_score": "Public score"}
    )
    st.bar_chart(
        chart,
        x="Submission",
        y=["Private score", "Public score"],
        y_label="Normalized weighted RMSLE",
        horizontal=True,
        stack=False,
        sort=False,
        height=300,
        alt="Public and private Kaggle scores for the five late submissions.",
    )
    st.dataframe(
        scores.drop(columns="local_file"),
        hide_index=True,
        column_config={
            "kaggle_submission": "Uploaded filename",
            "private_score": st.column_config.NumberColumn(
                "Private score", format="%.5f"
            ),
            "public_score": st.column_config.NumberColumn(
                "Public score", format="%.5f"
            ),
            "provenance": "Notebook evidence",
        },
    )
    best_score = scores["private_score"].min()
    st.markdown(
        f"**The lowest reported private score was {best_score:.5f}.** The exact "
        "feature recipe for that XGBoost upload is unavailable. These scores "
        "describe historical submissions and cannot establish the contribution "
        "of individual features."
    )
    st.link_button("Kaggle evaluation", f"{COMPETITION}/overview/evaluation")
    st.caption(
        "Both rolling notebooks sorted test rows before assigning predictions "
        "to template IDs by position. The audit found a row-alignment defect."
    )


def show_deployment() -> None:
    """Explain the locally served app and show its saved forecast output."""
    st.header("Deployment")
    st.markdown(
        "This local Streamlit app serves prepared charts, Kaggle scores, and a "
        "saved forecast. The notebook workflows handle training offline."
    )
    offline, files, app = st.columns(3)
    with offline.container(border=True):
        st.markdown("**Offline model runs**  \nFit models and export submission CSVs.")
    with files.container(border=True):
        st.markdown("**Prepared artifacts**  \nSave scores and small chart excerpts.")
    with app.container(border=True):
        st.markdown("**Streamlit app**  \nRead the saved files and render the views.")
    st.subheader("Saved basic LightGBM forecast · store 1, item 211203")
    st.caption(
        "August 16–31, 2017 · Earlier basic run, joined to Kaggle test dates by "
        "ID. The test set has no observed unit sales."
    )
    show_sales_chart("example_forecast.csv", "Forecast unit sales", "#3464AD")
    with st.expander("Run the app locally"):
        st.code("python -m streamlit run app.py", language="bash")


def main() -> None:
    """Render the portfolio presentation using only the prepared artifacts."""
    st.set_page_config(page_title="Favorita Grocery Sales Forecasting", layout="wide")
    st.title("Favorita Grocery Sales Forecasting")
    st.markdown(
        "A case study in forecasting how many units each Favorita store would sell "
        "of each item over a 16-day horizon."
    )
    period, models, volume = st.columns(3)
    period.metric("Forecast horizon", "16 days")
    models.metric("Model families", "LightGBM and XGBoost")
    volume.metric("Forecast rows", "3,370,464")

    sections = (
        show_cleaning, show_exploration, show_features, show_models,
        show_results, show_deployment,
    )
    for tab, show_section in zip(st.tabs(STAGES), sections, strict=True):
        with tab:
            show_section()


if __name__ == "__main__":
    main()

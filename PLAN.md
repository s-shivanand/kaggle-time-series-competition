# Favorita portfolio implementation plan

## Goal

Preserve the notebook research, make the two basic runs reproducible from Python, and present the original Kaggle results in a Streamlit app for portfolio visitors. See [AGENTS.md](AGENTS.md) for permanent rules and [docs/BASELINE.md](docs/BASELINE.md) for audited methods and artifact evidence.

## Phases and status

| Phase | Status | Deliverable and dependency |
|---|---|---|
| Audit and baseline | Complete | Document all notebook recipes, both estimators, the two Kaggle screenshots, and local artifact hashes. Preserve existing notebook files and local deletions. |
| Basic workflow extraction | Complete | Root-invoked Python command with separate data loading, preprocessing, features, training, prediction, and artifact evaluation modules. Both basic runs fit and export on synthetic data. |
| Portfolio app and documentation | Complete | Show six visual stages from data cleaning to local deployment, including all five reported scores and the saved forecast excerpt. The app starts with only its app dependencies and prepared artifacts. |
| Repeatable verification | Local checks passed | Model and app requirements install in Python 3.14.4. Synthetic tests and the Streamlit smoke test pass locally. GitHub Actions is configured. The full-data replay is approval-gated. |

## Design and acceptance criteria

- `src.run` supports `basic-lightgbm` and `basic-xgboost`. Keep feature order, casts, imputation, seed, and model settings from the corresponding notebook. Fit once, clip predictions, verify test IDs against the sample template, and write a new CSV without overwriting saved results. Reject training data dated after the fixed forecast origin.
- Keep the exploratory work and enriched runs in the notebook record. Tests illustrate observed-row lags, same-row target inclusion, and test sorting. Changes to the archived workflows require a separate methodology decision.
- Store scores and the two small series under `artifacts/`. Streamlit must run without Kaggle files or model dependencies, provide a visual for each lifecycle stage, show all five precise score pairs and their provenance, distinguish sales history from the basic LightGBM forecast, and handle an unavailable excerpt.
- Run `python -m unittest discover -s tests -v` in a clean Python 3.14.4 environment. Synthetic inputs exercise the cutoff, dtype and categorical behavior, original model settings and both fitted pipelines, prediction clipping, ID order, and safe output writing. The app smoke test runs from a directory containing only the app and prepared review artifacts.
- If a full-data replay is approved, validate the new LightGBM file's IDs exactly, then report maximum absolute and relative differences against local `submission_lightgbm.csv` and record the environment. Compare basic XGBoost with its saved notebook preview. Keep the Kaggle XGBoost upload's different provenance explicit.

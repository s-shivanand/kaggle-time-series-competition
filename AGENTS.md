# Project rules

- Use one agent. The original notebooks are the source of truth for the forecasting methods. Preserve them and their saved outputs as research records.
- Preserve model families, algorithms, data processing, features, training and validation, explicit parameters, random seeds, evaluation, and submission semantics. Keep original results as close as the original environment permits. Score improvements and new forecasting methods are outside scope.
- Use only data available at the forecast origin, including future-dated inputs explicitly supplied in Kaggle's test file. Keep historical experiments with known timing, target, or submission defects in the research record. Document discrepancies and obtain approval before making any methodology correction.
- Keep provenance for saved results. Report Kaggle scores as late-submission results, and distinguish verified artifact matches from uncertain recipe or upload mappings. Obtain approval before expensive training or full-data replay.
- Keep the repository runnable without packaging. Use small typed functions, clear docstrings, useful errors, explicit dependencies, and lightweight automated tests. Keep the Streamlit app independent of model training and the raw competition files.

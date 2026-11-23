# Original Favorita forecasting baseline

The project predicts daily unit sales for each store–item pair in the Corporación Favorita Kaggle competition. The fixed forecast origin is August 15, 2017. The 16-day forecast covers August 16–31, 2017 and contains 3,370,464 rows. [Competition overview](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting)

The audit used the notebooks at `1bf9a90`, saved notebook outputs, local ignored data and submissions, and two user-provided Kaggle submissions screenshots dated November 23, 2026. It required no full-data training. In the tables below, a preview match confirms only the values visible in the notebook output.

## Data and research sequence

The eight competition CSVs are stored locally under `data/raw/favorita-grocery-sales-forecasting/`.

| File | Contents and original use |
|---|---|
| `train.csv` | 125,497,040 labeled sales observations from January 1, 2013 to August 15, 2017. Columns are `id`, `date`, `store_nbr`, `item_nbr`, `unit_sales`, and `onpromotion`. |
| `test.csv` | 3,370,464 forecast rows with the same identifiers, dates, and promotion flags. Unit sales are withheld. |
| `sample_submission.csv` | The `id,unit_sales` template. IDs match raw test order. |
| `stores.csv` | 54 stores with city, state, type, and cluster. Left-joined on `store_nbr` in enriched runs. |
| `items.csv` | 4,100 items with family, class, and perishability. Left-joined on `item_nbr` in enriched runs. |
| `oil.csv` | 1,218 daily records. `dcoilwtico` is left-joined on date in enriched runs. |
| `holidays_events.csv` | 350 holiday and event records inspected in EDA. |
| `transactions.csv` | 83,488 store-date transaction counts inspected in EDA. |

Each fitted notebook reads the raw files independently. Git history shows this research progression.

1. [Data exploration](../notebooks/SKS_data_exploration.ipynb) and [outlier inspection](../notebooks/SKS_outliers.ipynb) inspect sizes, dtypes, sales summaries, missingness, identifier frequencies, and joins to stores and items. Saved results show zero duplicate `(date, store_nbr, item_nbr)` sales keys, 21,657,651 missing promotion values in the full history, and 43 missing oil prices. The 2017 training subset shows no missing promotion values. The outlier notebook contains descriptive checks.
2. [Basic LightGBM](../notebooks/SKS_build_time_data.ipynb) is the earliest saved modeling workflow. The [basic XGBoost](../notebooks/SKS_xgboost.ipynb) and [enriched XGBoost](../notebooks/SKS_xgboost_merge_features.ipynb) notebooks follow.
3. [Rolling XGBoost](../notebooks/SKS_xgboost_moving_avg.ipynb) and [rolling LightGBM](../notebooks/SKS_xgboost_moving_avg_minimal.ipynb) contain the current lag and rolling-feature experiments. The filename of the latter predates its LightGBM estimator. An earlier Git revision of the rolling XGBoost notebook contains different code, so use the audited revision when documenting this configuration.

`pls_understand_csv.ipynb` was already deleted in the working tree. Its archived copy is available with `git show 1bf9a90:notebooks/pls_understand_csv.ipynb`. It uses a machine-specific path, scans in 500,000-row chunks, takes the first 100,000 observations on or after August 15, 2016, and selects the first 100 distinct store–item keys. It scans the date-filtered data again for those keys and explores per-pair summary statistics and row-based lag-1 and lag-7 autocorrelation. It uses a one-row shift for 7- and 28-row rolling means. The saved correlations with sales are 0.608327 and 0.584137, respectively, on this selected data. The plotted example has 362 observations for store 1, item 211203, although its title says item 105575. Its saved lag-1 and lag-7 autocorrelations are 0.102156 and 0.118836. These are descriptive sample results.

## Training and submission behavior

The five fitted notebooks read `train.csv` in million-row chunks, retain 23,808,261 rows dated from January 1, 2017 onward, and fit once on all retained history through August 15. They drop `id` from the feature matrix, cast store to `int8`, item to `int32`, and target to `float32`, map Boolean promotions to 0/1, and extract year/month/day before dropping the date. All use most-frequent categorical imputation and ordinal encoding with `-1` for unknown categories. The numeric transformer precedes the categorical transformer in the pipeline.

The basic categorical columns are store and item IDs. The basic numeric columns are promotion, year, month, and day. Enriched runs add city, state, store type, and item family as categorical columns, and class, cluster, perishability, and oil price as numeric columns. They cast training year and class to `int16`, month/day/cluster/perishable to `int8`, and oil and rolling means to `float32`. The notebook applies the oil cast to the training frame again in the enriched XGBoost test cell. Rolling variants use `float32` for test promotion. The basic runs instead cast training year to `int8`, turning 2017 into `-31`, and leave the test year at 2017.

| Notebook | Active estimator and run-specific behavior | Notebook export |
|---|---|---|
| `SKS_build_time_data` | Basic LightGBM, numeric mean imputation, original target, no fitted lags. | `submission_lightgbm.csv` |
| `SKS_xgboost` | Basic XGBoost at `0.3`, numeric mean imputation, clips negative targets to zero, no fitted lags. | `submission_xgboost_lr_0_1.csv` |
| `SKS_xgboost_merge_features` | Enriched XGBoost at `0.3`, numeric mean imputation, no fitted lags. | `submission_xgboost_with_sample_weights_log_transform.csv` |
| `SKS_xgboost_moving_avg` | Enriched XGBoost at `0.03`, numeric imputation with 0, lag 16, and a 7-row mean of lag-16 values. | `submission_xgboost_lag16_mean7.csv` |
| `SKS_xgboost_moving_avg_minimal` | Enriched LightGBM, numeric imputation with 0, lags 3/7/14 and unshifted rolling means over 3/7/14/21 rows. | `lightgbm_lag3_7_14_21_with_features.csv` |

Both LightGBM runs set `n_estimators=1000`, `learning_rate=0.03`, `random_state=42`, and `verbose=10`. All three XGBoost runs set `n_estimators=1000`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method="hist"`, `objective="reg:squarederror"`, `eval_metric="rmse"`, `n_jobs=-1`, and `random_state=42`. Learning rates are in the table. The notebook leaves all other estimator settings at library defaults. Random Forest and alternative estimators present only in comments are excluded.

The enriched runs use `log1p(max(0, unit_sales))` and fit the estimator with sample weight `1 + 0.25 * perishable`. Rolling features are computed after sorting each store–item group by date. Lags are counts of recorded rows, and rolling means use `min_periods=1`. For inference, the rolling notebooks append the test rows with missing targets to training rows on or after the maximum training date minus 60 days, recompute features, and retain dates after the training maximum.

The notebooks assign predictions to the sample template by position. Basic runs clip predictions at zero. Enriched runs clip in log space and apply `expm1`. They write without a CSV index. The basic XGBoost notebook's preview gives `4.097089` for the first five predictions before clipping. The locally saved `submission_xgboost.csv` begins with `3.8672884` repeated five times.

## Evaluation and saved submissions

No fitted notebook uses a local holdout, validation dates, a backtest, or a local competition scorer. The XGBoost RMSE setting has no evaluation set. Kaggle computes normalized weighted root mean squared logarithmic error, with weights 1.25 for perishable items and 1.00 otherwise. The screenshots show five completed late submissions and unchecked selection boxes. [Official Kaggle evaluation](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/overview/evaluation)

| Kaggle filename | Private | Public | Local and notebook evidence |
|---|---:|---:|---|
| `submission_lightgbm.csv` | 1.48557 | 1.45230 | Local filename and saved basic LightGBM notebook preview agree. |
| `submission_xgboost.csv` | 1.47656 | 1.44706 | Local filename matches Kaggle. The basic XGBoost notebook's preview and export filename differ. |
| `xgboost_lag3_7_14_with_features.csv` | 1.19868 | 1.17484 | The local candidate is `xgboost_lag3_7_14.csv`. The filenames differ and the exact notebook recipe is unavailable. |
| `xgboost_mean_lag3_7_14_with_features.csv` | **1.09747** | **1.11674** | Local filename matches Kaggle. The author specified the minimal rolling notebook with XGBoost enabled and the 21-observation mean omitted. Full-file replay remains unverified. |
| `lightgbm_mean_lag3_7_14_21_with_features.csv` | 1.11764 | 1.13637 | The project author confirmed `SKS_xgboost_moving_avg_minimal.ipynb` as the source. The first five inverse-transformed preview values match within rounding precision. Full-file replay remains unverified. |

The lowest reported private score is 1.09747. The results do not provide a controlled comparison of feature contributions. Each local submission has the expected columns, 3,370,464 IDs in sample order, and finite nonnegative values. The SHA-256 hashes identify the audited local bytes. Matching a filename or preview does not establish that the local file is byte-identical to the upload.

The rolling LightGBM notebook keeps lags of 3, 7, and 14 observations and rolling means of 3, 7, 14, and 21 observations. Its final cell exports `lightgbm_lag3_7_14_21_with_features.csv` under the raw-data directory. The saved submission uses the `lightgbm_mean_lag3_7_14_21_with_features.csv` filename in `data/submission/`. The maximum absolute difference across the first five saved predictions and the inverse-transformed notebook preview is about `1.14e-6`. The recorded Kaggle score applies to the original upload. The [reproduction notebook](../notebooks/01_lightgbm.ipynb) and [command-line workflow](#command-line-runs) call the same Python modules and can compare every output row with the saved artifact. Both produced CSVs identical to the original notebook's output on a small fixture.

The [XGBoost reproduction notebook](../notebooks/02_xgboost.ipynb) uses the same Python modules with the commented XGBoost configuration from `SKS_xgboost_moving_avg_minimal.ipynb` and omits the 21-observation mean on both sides. Its 3-, 7-, and 14-observation lags and means differ from the archived lag-16 experiment in `SKS_xgboost_moving_avg.ipynb`. It matched the minimal rolling notebook with the author-specified edits on a fixture. Agreement with the archived competition CSV remains unverified.

| Local file in `data/submission/` | SHA-256 |
|---|---|
| `submission_lightgbm.csv` | `2cda2e23eb3444517c307ccd1ea593ca1621ce758dede4adb32516690afbc880` |
| `submission_xgboost.csv` | `9b32a60925ecaa98b9edf47cb1e8cf7a2c1d5195e8b19b89b83a762354d7558f` |
| `xgboost_lag3_7_14.csv` | `2c01334313caa18babbbaa7dbf6a69f8d1097ee9c7b70820ca98a967c210e624` |
| `xgboost_mean_lag3_7_14_with_features.csv` | `75fa84fbdeb1db6ed555e07235a5090711202770cabb45345aaf84dcc5c9049d` |
| `lightgbm_mean_lag3_7_14_21_with_features.csv` | `9f7a10f8743aa9b100191f97b21f8a246120789a5af090af486609b29f4cf7f0` |

## Limitations and environment

- The two rolling notebooks discard test IDs, sort rows by store/item/date, and assign predictions to the original sample order. Sorting changes 3,370,462 of 3,370,464 positions. The template IDs remain valid even when predictions are attached to the wrong rows.
- The minimal rolling training averages include the current target, a behavior preserved by both reproduction notebooks. The corresponding test targets are missing, so the two sides also have different feature availability.
- All enriched notebooks join oil by forecast date, including 12 actual oil prices dated after the fixed August 15 origin. The enriched variants remain historical experiments. The two basic workflows use calendar, identifier, and test-supplied promotion inputs.
- The local Darwin arm64 environment uses Python 3.14.4, pandas 3.0.5, NumPy 2.5.2, scikit-learn 1.9.0, XGBoost 3.4.0, LightGBM 4.7.0, and Matplotlib 3.11.1. The modeling notebooks declare Python 3.14.4 and the historical EDA declares Python 3.13.14. The local package versions are observed environment evidence. The original runs have no tracked dependency lockfile or fitted model artifact.
- The training CSV is 4.7 GB on disk. Chunked reading limits the input buffer, but the modeling notebooks concatenate the retained rows and fit in memory. An approved full replay should record its environment and numerical differences from the applicable saved evidence.

## Command-line runs

Run these commands from the repository root after extracting the Kaggle CSVs to `data/raw/favorita-grocery-sales-forecasting/`.

### Basic models

```bash
uv run --locked --no-group app python -m src.run --run basic-lightgbm \
  --data-dir data/raw/favorita-grocery-sales-forecasting \
  --output data/reproduction/reproduced_lightgbm.csv
```

Use `--run basic-xgboost` and a new output filename for the basic XGBoost run. On macOS the model libraries may need the OpenMP runtime.

### Rolling LightGBM

```bash
uv run --locked --exact --no-group app python -m src.replay_rolling_lightgbm \
  --data-dir data/raw/favorita-grocery-sales-forecasting \
  --output data/reproduction/lightgbm_mean_lag3_7_14_21_with_features.csv \
  --reference data/submission/lightgbm_mean_lag3_7_14_21_with_features.csv
```

The optional reference check reports exact matches, matches within `rtol=1e-6` and `atol=1e-6`, and the mean and maximum absolute prediction differences. Choose a fresh output path for each run.

### Tests and dashboard data

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 uv run --locked python -m unittest discover -s tests -v
```

Regenerate the dashboard chart excerpts from the source history and the saved basic LightGBM submission.

```bash
uv run --locked --no-default-groups python -m src.prepare_visualization_data \
  --data-dir data/raw/favorita-grocery-sales-forecasting \
  --submission data/submission/submission_lightgbm.csv \
  --output-dir artifacts
```

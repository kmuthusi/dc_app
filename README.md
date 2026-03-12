# Dixon-Coles Football Model Pipeline

This repository contains a league-by-league Dixon-Coles workflow for:

- ingesting match data from Excel workbooks,
- training one model artifact per league,
- generating single or batch predictions from saved artifacts,
- running rolling backtests to measure predictive performance, and
- serving a prediction-focused Streamlit UI.

## Current Project Structure

- `train_models.py`: builds the master dataset and trains one `.joblib` artifact per league.
- `predict_matches.py`: runs batch predictions from a saved artifact and a fixtures file.
- `backtest_models.py`: runs rolling out-of-sample backtests and compares model probabilities with bookmaker implied probabilities when odds exist.
- `app.py`: Streamlit prediction UI for single-match, one-league batch, and multi-league batch forecasts.
- `src/ingest.py`: ingestion and dataset standardization helpers.
- `src/dixon_coles.py`: Dixon-Coles model fitting, prediction, and artifact serialization helpers.
- `data/`: input workbooks.
- `models/`: trained league artifacts such as `E0.joblib`, `E1.joblib`, and `D1.joblib`.
- `processed/`: saved unified dataset and training report.

## Installation

```bash
pip install -r requirements.txt
```

## Data Inputs

The training and backtesting scripts use the same data sources.

### `data/euro/all-euro-data-<season>.xlsx`

- Each workbook can contain multiple league sheets such as `E0`, `D1`, or `SP1`.
- Known football-data columns are standardized.
- Extra columns are preserved in the master dataset rather than dropped.
- Required result fields are the home team, away team, and full-time goals.

### `data/new_leagues_data.xlsx`

- Optional supplementary workbook.
- Each sheet is treated as a league.
- Useful when a competition is not covered in the `all-euro-data-*` files.

## Core Outputs

- `models/<league_id>.joblib`: serialized model artifact for each trained league.
- `processed/master_matches.parquet` or `processed/master_matches.csv`: unified dataset.
- `processed/train_report.json`: training summary across leagues.
- `backtests/<league_id>/...`: rolling backtest outputs such as per-match predictions, calibration tables, and best-parameter summaries.

## Training Models

Train all available leagues and save artifacts:

```bash
python train_models.py `
  --all-euro-dir data/euro `
  --new-leagues-file data/new_leagues_data.xlsx `
  --out-dir models `
  --processed-out processed/master_matches.parquet `
  --report-out processed/train_report.json `
  --last-n-seasons 3 `
  --half-life-days 180 `
  --max-goals 10 `
  --l2 0.001 `
  --xg-weight 0.0
```

Notes:

- `--half-life-days 0` disables time decay.
- `--xg-weight` only has an effect when the unified dataset contains `home_xg` and `away_xg`.
- The processed dataset writer falls back to CSV if parquet writing fails.

## Making Predictions From an Artifact

### CLI batch prediction

```bash
python predict_matches.py `
  --artifact models/E0.joblib `
  --fixtures fixtures.csv `
  --home-col HomeTeam `
  --away-col AwayTeam `
  --out predictions.csv
```

The fixtures file can be CSV or Excel. The predictor only requires team-name columns.

Required fields for a one-league prediction file:

- one home-team column such as `HomeTeam`
- one away-team column such as `AwayTeam`

Minimal example:

```csv
HomeTeam,AwayTeam
Arsenal,Chelsea
Liverpool,Everton
```

You do not need to provide market odds to make predictions.

### Mixed-league batch prediction

If one fixtures file contains matches from multiple leagues, include a league column
such as `Div` and point the predictor at the whole artifacts directory:

```bash
python predict_matches.py `
  --artifacts-dir models `
  --fixtures mixed_fixtures.csv `
  --league-col Div `
  --home-col HomeTeam `
  --away-col AwayTeam `
  --threshold-map backtests_threshold_sweep/best_exp_thresholds_all.json `
  --out mixed_predictions.csv
```

How this works:

- each row reads its league from `--league-col`,
- the script loads `models/<league_id>.joblib`,
- predictions are generated with the correct league model,
- if `--threshold-map` is supplied, `exp_outcome` uses the best stored draw threshold for that league.

This is the recommended path when your batch contains leagues such as `E0`, `E1`, `D1`, and `SP1` in the same file.

Required fields for a multi-league prediction file:

- one league column such as `Div`
- one home-team column such as `HomeTeam`
- one away-team column such as `AwayTeam`

The `Div` values must match artifact names in `models/`, such as `E0`, `E1`, `D1`, or `SC0`.

Minimal example:

```csv
Div,HomeTeam,AwayTeam
E0,Arsenal,Chelsea
SC0,Celtic,Rangers
```

Extra columns are allowed in either file type and will be ignored unless you explicitly map them in the UI.

### Sample fixtures

The repository includes a few small fixture files in the project root so you can
test prediction flows quickly without preparing your own input first:

- `fixtures.csv`: simple one-league example
- `fixtures_test.csv`: mixed-league batch example
- `D1-fixtures.csv`, `E1-fixtures.csv`, `SC0-fixtures.csv`, `sp_fixtures.csv`: league-specific examples

These are kept in the repo root for now to match the existing command examples.
If the project grows, they can be moved later into an `examples/` or
`sample_data/` folder with the README commands updated accordingly.

### Prediction output columns

Prediction tables contain:

- `league_id`
- `home_team`, `away_team`
- `exp_home_goals`, `exp_away_goals`
- `exp_outcome` derived from expected goals using a draw cutoff threshold
- `pred_home_goals`, `pred_away_goals`
- `pred_outcome` derived from the most likely exact scoreline
- `p_home`, `p_draw`, `p_away`
- `fair_odds_home`, `fair_odds_draw`, `fair_odds_away`

If `--top-k-scores` is used, the output also includes:

- `most_likely_score`
- `p_most_likely_score`
- `top_scorelines`

## Streamlit UI

Launch the UI with:

```bash
streamlit run app.py
```

The current UI is prediction-only. It does not train models.

Features:

- sidebar selection of division and matching artifact,
- single-match prediction tab,
- one-league batch-prediction tab for uploaded fixtures,
- multi-league batch-prediction tab that routes each row by league code such as `Div`,
- user-guide tab,
- mobile-friendly responsive spacing,
- a persistent footer with dynamic year, version, and organization branding,
- automatic restriction of single-match team dropdowns to the latest season when `processed/master_matches.*` is available,
- automatic use of `backtests_threshold_sweep/best_exp_thresholds_all.json` when present so `exp_outcome` can use league-specific draw thresholds in the multi-league tab.

### Deployment note

If you deploy the app from this repository exactly as committed, the hosted app will
start, but it will not have bundled league artifacts because `models/` is ignored in git.

- single-match and one-league batch prediction still work if the user uploads a `.joblib` artifact in the sidebar,
- multi-league batch prediction requires the deployment to include `models/<league_id>.joblib` files,
- if the deployed sidebar shows no divisions, that means no local artifacts were shipped with the app.

If you want full hosted functionality without manual artifact upload, you need to publish
the required model files with the deployment or download them from a separate storage location at runtime.

## Backtesting and Performance Measurement

Use `backtest_models.py` to measure out-of-sample model quality.

Example:

```bash
python backtest_models.py `
  --all-euro-dir data/euro `
  --new-leagues-file data/new_leagues_data.xlsx `
  --league-id E0 `
  --out-dir backtests `
  --last-n-seasons 3 `
  --half-life-days-grid 180 `
  --xg-weight-grid 0 `
  --draw-threshold-grid 0.10,0.15,0.20,0.25,0.30,0.35 `
  --retrain-every 50
```

The backtest script writes:

- per-match prediction files,
- calibration tables,
- `summary.csv` and `summary.json`,
- `best_params.json` for each league,
- `best_params_all.json` across leagues,
- `best_exp_threshold.json` for each league,
- `best_exp_thresholds_all.json` across leagues.

### How to judge project performance

The main performance files are produced by `backtest_models.py`:

- `summary.csv`: one row per league and parameter combination.
- `<league>/best_params.json`: best setting for that league by model log loss.
- `<league>/best_exp_threshold.json`: best draw threshold for `exp_outcome` by `exp_accuracy`.
- `<league>/*_predictions.csv`: per-match predictions with true outcomes and errors.
- `<league>/*_calibration.csv`: reliability of predicted top probabilities.

The most useful metrics are:

- `model_logloss`: lower is better.
- `model_brier`: lower is better.
- `model_accuracy`: higher is better.
- `exp_accuracy`: higher is better; this measures the accuracy of `exp_outcome`, where draws are assigned when `abs(exp_home_goals - exp_away_goals)` is below the configured draw threshold.
- `book_logloss`, `book_brier`, `book_accuracy`: bookmaker benchmark when odds exist.

If your model beats bookmaker metrics on held-out matches, that is strong evidence the specification is competitive. If bookmaker log loss is lower than model log loss, the market is still better calibrated than the model on that evaluation set.

## Smoke Test Result

A real smoke test was run in this workspace on `E0` using:

```bash
python backtest_models.py `
  --all-euro-dir data/euro `
  --new-leagues-file data/new_leagues_data.xlsx `
  --league-id E0 `
  --out-dir backtests_smoke `
  --last-n-seasons 2 `
  --half-life-days-grid 180 `
  --xg-weight-grid 0 `
  --retrain-every 100 `
  --min-train-matches 20
```

This produced:

- `n_pred = 263`
- `model_logloss = 1.0407`
- `model_brier = 0.6269`
- `model_accuracy = 0.5057`
- `book_logloss = 1.0132`
- `book_brier = 0.6085`
- `book_accuracy = 0.5017`

Interpretation: on that smoke test, the model was slightly better than the bookmaker on raw accuracy, but worse on log loss and Brier score, which means the bookmaker probabilities were better calibrated overall.

## Current Known Gaps

- `streamlit` must be installed in the active environment before running the UI.
- Performance should be judged from rolling backtests rather than training fit alone.

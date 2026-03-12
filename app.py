import io
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.dixon_coles import DixonColesModel, model_from_artifact


APP_VERSION = "1.1.0"
EXP_OUTCOME_DRAW_THRESHOLD = 0.25


st.set_page_config(page_title="Dixon–Coles Prediction App", layout="wide")
# prediction-only: no training controls

st.markdown(
    """
    <style>
    .block-container {
        padding-bottom: 6rem;
    }
    .xalec-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        z-index: 999999;
        padding: 0.75rem 1rem;
        background: rgba(8, 16, 24, 0.96);
        color: #f3f6fb;
        border-top: 2px solid #00b894;
        box-shadow: 0 -10px 24px rgba(0, 0, 0, 0.22);
        backdrop-filter: blur(8px);
    }
    .xalec-footer__inner {
        max-width: 1200px;
        margin: 0 auto;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .xalec-footer__meta {
        color: #9fe7d6;
        font-weight: 600;
    }
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.9rem;
            padding-right: 0.9rem;
            padding-bottom: 7.5rem;
        }
        .xalec-footer {
            padding: 0.85rem 0.9rem;
        }
        .xalec-footer__inner {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.25rem;
            font-size: 0.88rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    data = uploaded_file.getbuffer()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


def _expected_goal_outcome(exp_home_goals: float, exp_away_goals: float, draw_threshold: float = EXP_OUTCOME_DRAW_THRESHOLD) -> str:
    diff = float(exp_home_goals) - float(exp_away_goals)
    if abs(diff) <= float(draw_threshold):
        return "D"
    return "H" if diff > 0 else "A"


def _load_threshold_map(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    out: dict[str, float] = {}
    for league_id, payload in raw.items():
        if isinstance(payload, dict) and "draw_threshold" in payload:
            out[str(league_id)] = float(payload["draw_threshold"])
    return out


def _resolve_draw_threshold(league_id: str, threshold_map: dict[str, float], default_threshold: float = EXP_OUTCOME_DRAW_THRESHOLD) -> float:
    return float(threshold_map.get(str(league_id), default_threshold))


def _predict_rows_multileague(
    models_dir: Path,
    fixtures: pd.DataFrame,
    league_col: str,
    home_col: str,
    away_col: str,
    top_k: int = 0,
    on_unknown: str = "nan",
    threshold_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    out_rows = []
    model_cache: dict[str, tuple[dict, DixonColesModel]] = {}
    threshold_map = threshold_map or {}

    leagues = fixtures[league_col].astype(str).str.strip()
    homes = fixtures[home_col].astype(str).str.strip()
    aways = fixtures[away_col].astype(str).str.strip()

    for league_id, home_team, away_team in zip(leagues, homes, aways):
        artifact_path = models_dir / f"{league_id}.joblib"
        if not artifact_path.exists():
            if on_unknown == "error":
                raise FileNotFoundError(f"Artifact not found for league_id={league_id!r}: {artifact_path}")
            out_rows.append(
                {
                    "league_id": league_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "draw_threshold": np.nan,
                    "exp_home_goals": np.nan,
                    "exp_away_goals": np.nan,
                    "exp_outcome": np.nan,
                    "pred_home_goals": np.nan,
                    "pred_away_goals": np.nan,
                    "pred_outcome": np.nan,
                    "p_home": np.nan,
                    "p_draw": np.nan,
                    "p_away": np.nan,
                    "fair_odds_home": np.nan,
                    "fair_odds_draw": np.nan,
                    "fair_odds_away": np.nan,
                    "note": "artifact_missing",
                }
            )
            continue

        if league_id not in model_cache:
            artifact = joblib.load(artifact_path)
            model_cache[league_id] = (artifact, model_from_artifact(artifact))

        artifact, model = model_cache[league_id]
        resolved_league_id = str(artifact.get("league_id", league_id))
        draw_threshold = _resolve_draw_threshold(resolved_league_id, threshold_map)
        pred = _predict_rows(
            model,
            resolved_league_id,
            pd.Series([home_team]),
            pd.Series([away_team]),
            top_k=top_k,
            on_unknown=on_unknown,
        )
        pred.insert(1, "draw_threshold", round(draw_threshold, 3))
        pred.loc[:, "exp_outcome"] = pred.apply(
            lambda r: _expected_goal_outcome(r["exp_home_goals"], r["exp_away_goals"], draw_threshold)
            if pd.notna(r["exp_home_goals"]) and pd.notna(r["exp_away_goals"]) else np.nan,
            axis=1,
        )
        pred["note"] = pred.get("note", "")
        out_rows.append(pred.iloc[0].to_dict())

    return pd.DataFrame(out_rows)


def _predict_rows(model, league_id: str, home: pd.Series, away: pd.Series, top_k: int = 0, on_unknown: str = "nan") -> pd.DataFrame:
    out_rows = []
    for h, a in zip(home.astype(str).str.strip(), away.astype(str).str.strip()):
        try:
            lam_raw, nu_raw = model.expected_goals(h, a)
            probs = model.outcome_probs(h, a)
            # rounding
            lam = round(lam_raw, 2)
            nu = round(nu_raw, 2)
            ph = round(probs["H"], 3)
            pd_raw = round(probs["D"], 3)
            pa = round(probs["A"], 3)

            # exact score and pred_outcome remain based on the score-matrix argmax.
            Pmat = model.score_matrix(h, a)
            idx = int(np.argmax(Pmat))
            home_pred = int(idx // Pmat.shape[1])
            away_pred = int(idx % Pmat.shape[1])
            outcome = "H" if home_pred > away_pred else "D" if home_pred == away_pred else "A"
            exp_outcome = _expected_goal_outcome(lam_raw, nu_raw)

            row = {
                "league_id": league_id,
                "home_team": h,
                "away_team": a,
                "exp_home_goals": lam,
                "exp_away_goals": nu,
                "exp_outcome": exp_outcome,
                "pred_home_goals": home_pred,
                "pred_away_goals": away_pred,
                "pred_outcome": outcome,
                "p_home": ph,
                "p_draw": pd_raw,
                "p_away": pa,
                "fair_odds_home": round((1.0 / probs["H"]) if probs["H"] > 0 else np.nan, 3),
                "fair_odds_draw": round((1.0 / probs["D"]) if probs["D"] > 0 else np.nan, 3),
                "fair_odds_away": round((1.0 / probs["A"]) if probs["A"] > 0 else np.nan, 3),
            }

            if top_k and top_k > 0:
                P = model.score_matrix(h, a)
                flat = P.ravel()
                k = int(max(1, min(top_k, flat.size)))
                top_idx = np.argpartition(-flat, k - 1)[:k]
                top_idx = top_idx[np.argsort(-flat[top_idx])]

                scores = []
                for idx in top_idx:
                    x = int(idx // P.shape[1])
                    y = int(idx % P.shape[1])
                    scores.append({"score": f"{x}-{y}", "p": float(flat[idx])})

                row["most_likely_score"] = scores[0]["score"]
                row["p_most_likely_score"] = scores[0]["p"]
                row["top_scorelines"] = json.dumps(scores, ensure_ascii=False)

        except KeyError:
            if on_unknown == "error":
                raise
            row = {
                "league_id": league_id,
                "home_team": h,
                "away_team": a,
                "exp_home_goals": np.nan,
                "exp_away_goals": np.nan,
                "exp_outcome": np.nan,
                "p_home": np.nan,
                "p_draw": np.nan,
                "p_away": np.nan,
                "fair_odds_home": np.nan,
                "fair_odds_draw": np.nan,
                "fair_odds_away": np.nan,
            }
        out_rows.append(row)

    return pd.DataFrame(out_rows)


st.title("Dixon–Coles Prediction App")

# sidebar controls for division/artifact selection
st.sidebar.header("Prediction settings")
models_dir = Path("models")
available = []
if models_dir.exists():
    available = sorted({p.stem for p in models_dir.glob("*.joblib")})

if not available:
    st.sidebar.info(
        "No bundled model artifacts were found in models/. On a hosted deployment, "
        "this usually means the repo was deployed without tracked .joblib files. "
        "You can still use single-league prediction by uploading a model artifact below."
    )

div = st.sidebar.selectbox("Division (league_id)", [""] + available)
artifact = None
artifact_path = None

if div:
    # list artifacts matching division prefix
    candidates = [p for p in models_dir.glob(f"{div}*.joblib")] if models_dir.exists() else []
    if candidates:
        sel = st.sidebar.selectbox("Artifact file", [p.name for p in candidates])
        artifact_path = models_dir / sel
        artifact = joblib.load(artifact_path)
    else:
        st.sidebar.warning("No artifact for selected division in ./models")

# allow upload fallback
upl = st.sidebar.file_uploader("Or upload artifact", type=["joblib"], accept_multiple_files=False)
if upl:
    artifact = joblib.load(io.BytesIO(upl.getbuffer()))
    artifact_path = None

threshold_map_path = Path("backtests_threshold_sweep/best_exp_thresholds_all.json")
threshold_map = _load_threshold_map(threshold_map_path)

model = None
league_id = None
teams = []
teams_current: list[str] = []

if artifact:
    model = model_from_artifact(artifact)
    league_id = artifact.get("league_id", "unknown")
    teams = sorted(model.fit_result.teams) if model.fit_result else []

    # attempt to load processed master matches to identify current season teams
    teams_current: list[str] = []
    proc_paths = [Path("processed/master_matches.parquet"), Path("processed/master_matches.csv")]
    for p in proc_paths:
        if p.exists():
            try:
                df_proc = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)
                sub = df_proc[df_proc["league_id"] == league_id]
                if not sub.empty and "season" in sub.columns:
                    last_season = sub["season"].max()
                    hrs = sub.loc[sub["season"] == last_season, "home_team"].astype(str)
                    ars = sub.loc[sub["season"] == last_season, "away_team"].astype(str)
                    teams_current = sorted(set(hrs.tolist() + ars.tolist()))
            except Exception:
                pass
            break

tab_single, tab_batch, tab_multi, tab_help = st.tabs(["Single Prediction", "Batch Predictions", "Multi-League Batch", "User Guide"])

with tab_single:
    if not artifact:
        st.info("Load a single league artifact from the sidebar to use single-match prediction.")
        if not available:
            st.caption("Deployment note: no local artifacts were found in models/, so upload a .joblib model in the sidebar to use this tab.")
    else:
        st.success(f"Loaded league: {league_id}  |  teams: {len(teams)}")
        if teams_current:
            st.caption(f"{len(teams_current)} teams found for most recent season")
        if artifact_path:
            st.caption(f"From: {artifact_path}")
        if div and div != league_id:
            st.warning(f"Artifact league_id {league_id} does not match selected division {div}")

        st.subheader("Single‑match prediction")
        pick_teams = teams_current if teams_current else teams
        c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
        with c1:
            home_team = st.selectbox("Home team", pick_teams)
        with c2:
            away_team = st.selectbox("Away team", pick_teams, index=1 if len(pick_teams) > 1 else 0)
        with c3:
            top_k = st.number_input("Top K scorelines", min_value=0, max_value=20, value=5)
        with c4:
            do_pred = st.button("Predict")

        if do_pred:
            pred_df = _predict_rows(model, league_id, pd.Series([home_team]), pd.Series([away_team]), top_k=int(top_k))
            st.dataframe(pred_df, use_container_width=True)

            if int(top_k) > 0 and "top_scorelines" in pred_df.columns:
                scores = json.loads(pred_df.loc[0, "top_scorelines"])
                st.write("Top scorelines")
                st.dataframe(pd.DataFrame(scores), use_container_width=False)

with tab_batch:
    if not artifact:
        st.info("Load a single league artifact from the sidebar to use the one-league batch predictor.")
        if not available:
            st.caption("Deployment note: if your hosted app does not include models/, upload a .joblib artifact in the sidebar first.")
    else:
        st.subheader("Batch prediction from file")
        fx_file = st.file_uploader("Upload fixtures CSV/XLSX", type=["csv", "xlsx", "xls"], accept_multiple_files=False)
        if fx_file is not None:
            # cache DataFrame in session state so it survives reruns
            if (
                "_fx_name" not in st.session_state
                or st.session_state._fx_name != fx_file.name
            ):
                try:
                    st.session_state._fx_df = _read_table(fx_file)
                    st.session_state._fx_name = fx_file.name
                except Exception as err:
                    st.error(f"Failed to read fixtures: {err}")
                    st.session_state._fx_df = None
            fx = st.session_state.get("_fx_df")
            if fx is not None:
                st.write("Fixtures preview")
                st.dataframe(fx.head(50), use_container_width=True)

                home_col = st.selectbox("Home column", fx.columns.tolist(), index=0)
                away_col = st.selectbox("Away column", fx.columns.tolist(), index=1 if len(fx.columns) > 1 else 0)
                top_k_batch = st.number_input("Top K scorelines (batch)", min_value=0, max_value=20, value=0)
                on_unknown = st.selectbox("Unknown teams", ["nan", "error"], index=0)

                if st.button("Run batch prediction"):
                    pred = _predict_rows(
                        model,
                        league_id,
                        fx[home_col],
                        fx[away_col],
                        top_k=int(top_k_batch),
                        on_unknown=on_unknown,
                    )
                    st.session_state._last_batch = pred
        # display previous results if any
        if "_last_batch" in st.session_state:
            st.write("Predictions")
            st.dataframe(st.session_state._last_batch, use_container_width=True)
            csv_bytes = st.session_state._last_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download predictions CSV",
                data=csv_bytes,
                file_name=f"predictions_{league_id}.csv",
                mime="text/csv",
            )

with tab_multi:
    st.subheader("Multi-league batch prediction")
    st.caption("Use a fixtures file with a league column such as Div. Each row will be matched to models/<league_id>.joblib.")
    if not available:
        st.warning(
            "Multi-league mode requires bundled artifacts in models/ on the deployed app. "
            "This mode will not run until the deployment includes files like models/E0.joblib, "
            "models/E1.joblib, and other required league artifacts."
        )
    multi_file = st.file_uploader(
        "Upload multi-league fixtures CSV/XLSX",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
        key="multi_league_file",
    )
    if multi_file is not None:
        if (
            "_multi_fx_name" not in st.session_state
            or st.session_state._multi_fx_name != multi_file.name
        ):
            try:
                st.session_state._multi_fx_df = _read_table(multi_file)
                st.session_state._multi_fx_name = multi_file.name
            except Exception as err:
                st.error(f"Failed to read fixtures: {err}")
                st.session_state._multi_fx_df = None

        multi_fx = st.session_state.get("_multi_fx_df")
        if multi_fx is not None:
            st.write("Fixtures preview")
            st.dataframe(multi_fx.head(50), use_container_width=True)

            cols = multi_fx.columns.tolist()
            div_default = cols.index("Div") if "Div" in cols else 0
            home_default = cols.index("HomeTeam") if "HomeTeam" in cols else min(1, len(cols) - 1)
            away_default = cols.index("AwayTeam") if "AwayTeam" in cols else min(2, len(cols) - 1)

            c1, c2 = st.columns(2)
            with c1:
                league_col = st.selectbox("League column", cols, index=div_default, key="multi_league_col")
                home_col = st.selectbox("Home column", cols, index=home_default, key="multi_home_col")
            with c2:
                away_col = st.selectbox("Away column", cols, index=away_default, key="multi_away_col")
                top_k_multi = st.number_input("Top K scorelines (multi-league)", min_value=0, max_value=20, value=0, key="top_k_multi")

            on_unknown_multi = st.selectbox("Missing artifact or unknown teams", ["nan", "error"], index=0, key="on_unknown_multi")
            if threshold_map:
                st.caption(f"Using per-league draw thresholds from {threshold_map_path}")
            else:
                st.caption(f"Threshold map not found. Using default draw threshold {EXP_OUTCOME_DRAW_THRESHOLD:.2f} for all leagues.")

            if st.button("Run multi-league batch prediction"):
                pred_multi = _predict_rows_multileague(
                    models_dir,
                    multi_fx,
                    league_col=league_col,
                    home_col=home_col,
                    away_col=away_col,
                    top_k=int(top_k_multi),
                    on_unknown=on_unknown_multi,
                    threshold_map=threshold_map,
                )
                st.session_state._last_multi_batch = pred_multi

    if "_last_multi_batch" in st.session_state:
        st.write("Predictions")
        st.dataframe(st.session_state._last_multi_batch, use_container_width=True)
        csv_bytes = st.session_state._last_multi_batch.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download multi-league predictions CSV",
            data=csv_bytes,
            file_name="predictions_multileague.csv",
            mime="text/csv",
        )

with tab_help:
        st.markdown(
            """
### How to use the prediction app

1. **Select a division** from the sidebar.  The artifact list updates to
   match the chosen league.
2. **Load a model** file (either choose an existing artifact or upload one).
   Once loaded, you can see how many teams are available and, if the
   processed dataset exists, teams from the most recent season.
3. Navigate to the **Single Prediction** tab for a one‑off forecast, the
    **Batch Predictions** tab for a one-league fixtures file, or the
    **Multi-League Batch** tab for one file containing rows from multiple leagues.
    Follow the on-screen prompts for column names and options.

### Required prediction file structure

For uploaded prediction files, the app accepts `.csv`, `.xlsx`, and `.xls`.

**One-league batch file**

- Required fields: one home-team column and one away-team column
- Recommended column names: `HomeTeam`, `AwayTeam`
- Example:

| HomeTeam | AwayTeam |
| --- | --- |
| Arsenal | Chelsea |
| Liverpool | Everton |

**Multi-league batch file**

- Required fields: one league column, one home-team column, and one away-team column
- Recommended column names: `Div`, `HomeTeam`, `AwayTeam`
- The league values must match artifact names in `models/`, for example `E0`, `E1`, `D1`, `SC0`
- Example:

| Div | HomeTeam | AwayTeam |
| --- | --- | --- |
| E0 | Arsenal | Chelsea |
| SC0 | Celtic | Rangers |

### Required fields

- `HomeTeam`: home team name exactly as used in the trained league artifact
- `AwayTeam`: away team name exactly as used in the trained league artifact
- `Div`: required only for multi-league files; must match the target artifact league id

Extra columns are allowed and will be ignored for prediction.

### Deployment note

- Single-match and one-league batch prediction can run on a hosted app if you upload a `.joblib` artifact in the sidebar.
- Multi-league batch prediction requires the deployment itself to include the `models/` directory with one artifact per league.
- If your deployed app shows no divisions in the sidebar, it means no bundled artifacts were shipped with the app.

You do **not** need to provide any market odds; the model only uses team
names and (optionally) expected‑goals data baked into the artifact.

For more details, see the project README or run `python predict_matches.py`
from the command line.
            """
        )

st.markdown(
    f"""
    <div class="xalec-footer">
        <div class="xalec-footer__inner">
            <div>Xalec AI Team</div>
            <div class="xalec-footer__meta">&copy; {datetime.now().year} | Dixon-Coles Prediction App | v{APP_VERSION}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


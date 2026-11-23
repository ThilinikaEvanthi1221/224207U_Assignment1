"""
Sri Lankan Full Blood Count (FBC) Dashboard - HB prediction with ensemble learning.
Run:  streamlit run app.py

Folder layout expected:
    app.py
    saved_models/        (*.pkl from your notebook + model_comparison.pkl, optional test_data.pkl)
    data/your_data.csv   (or any .csv next to app.py; you can also upload one in the sidebar)
"""
from pathlib import Path



import joblib
import numpy as np
import sys
import importlib.util

print("PYTHON USED BY STREAMLIT:", sys.executable)
print("PANDAS LOCATION:", importlib.util.find_spec("pandas").origin)
print("PYTHON PATH:", sys.path)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Sri Lankan FBC Dashboard", page_icon="🩸", layout="wide")

BASE = Path(__file__).parent
MODEL_DIR = BASE / "saved_models"
RANDOM_STATE = 42  # use the SAME value you used in the notebook split

MODEL_FILES = [
    "linear_regression", "random_forest", "extra_trees", "bagging_regressor",
    "xgboost", "lightgbm", "catboost", "stacking", "ann", "hybrid_rf", "hybrid_ann",
]
PRETTY = {
    "linear_regression": "Linear Regression (baseline)",
    "random_forest": "Random Forest (bagging)",
    "extra_trees": "Extra Trees (bagging)",
    "bagging_regressor": "Bagging Regressor",
    "xgboost": "XGBoost (boosting)",
    "lightgbm": "LightGBM (boosting)",
    "catboost": "CatBoost (boosting)",
    "stacking": "Stacking",
    "ann": "ANN (MLP)",
    "hybrid_rf": "Hybrid RF",
    "hybrid_ann": "Hybrid ANN (RF + ANN)",
}

from xgboost import XGBRegressor


class XGBNative:
    """XGBoost loaded from its native JSON file (no pickle)."""
    def __init__(self):
        self.pre = joblib.load(MODEL_DIR / "xgboost_preprocessor.pkl")
        self.model = XGBRegressor()
        self.model.load_model(str(MODEL_DIR / "xgboost_native.json"))
        self.feature_names_in_ = np.array(self.pre.feature_names_in_)

    def predict(self, X):
        return self.model.predict(self.pre.transform(X))


class StackingParts:
    """Stacking rebuilt from the saved parts."""
    def __init__(self):
        f = MODEL_DIR / "stacking_parts"
        self.pre = joblib.load(f / "preprocessor.pkl")
        self.final = joblib.load(f / "final.pkl")
        self.rf = joblib.load(f / "rf.pkl")
        self.lgbm = joblib.load(f / "lgbm.pkl")
        self.cat = joblib.load(f / "cat.pkl")
        self.xgb = XGBRegressor()
        self.xgb.load_model(str(f / "xgb.json"))
        self.feature_names_in_ = np.array(self.pre.feature_names_in_)

    def predict(self, X):
        Xt = self.pre.transform(X)
        base = np.column_stack([self.rf.predict(Xt), self.xgb.predict(Xt),
                                self.lgbm.predict(Xt), self.cat.predict(Xt)])
        return self.final.predict(base)

# ----------------------------------------------------------------- loading
@st.cache_resource
def load_models():
    loaded, errors = {}, {}
    for name in MODEL_FILES:
        try:
            if name == "xgboost":
                loaded[name] = XGBNative()
            elif name == "stacking":
                loaded[name] = StackingParts()
            else:
                path = MODEL_DIR / f"{name}.pkl"
                if not path.exists():
                    errors[name] = "File not found"
                    continue
                loaded[name] = joblib.load(path)
        except Exception as e:  # noqa: BLE001
            errors[name] = str(e)
    return loaded, errors


@st.cache_data
def load_pickle(path_str):
    return joblib.load(path_str)


@st.cache_data
def read_csv(path_str):
    return pd.read_csv(path_str)


models, load_errors = load_models()

def add_date_features(d):
    d = d.copy()
    date_col = next((c for c in d.columns if "date" in c.lower()), None)
    if date_col is not None:
        dt = pd.to_datetime(d[date_col], errors="coerce")
        for name, vals in {
            "year": dt.dt.year,
            "month": dt.dt.month,
            "quarter": dt.dt.quarter,
            "day_of_week": dt.dt.dayofweek,
        }.items():
            if name not in d.columns:
                d[name] = vals
    return d

# ----------------------------------------------------------------- sidebar
st.sidebar.title("🩸 FBC Dashboard")
st.sidebar.caption("HB prediction using ensemble learning")

csv_candidates = sorted(list(BASE.glob("*.csv")) + list((BASE / "data").glob("*.csv")))
df = None
# uploaded = st.sidebar.file_uploader("Upload dataset (CSV)", type="csv")
# if uploaded is not None:
#     df = pd.read_csv(uploaded)
if csv_candidates:
    choice = st.sidebar.selectbox("Dataset file", csv_candidates, format_func=lambda p: p.name)
    df = read_csv(str(choice))

if df is not None:
    if "parameter" in df.columns:
        df = df[df["parameter"].astype(str).str.strip().str.upper() == "HB"].copy()
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[(df["value"] >= 2) & (df["value"] <= 25)]
    df = add_date_features(df)

target = None
if df is not None:
    guess = next((c for c in df.columns if c.strip().upper() in {"HB", "HGB", "HAEMOGLOBIN", "HEMOGLOBIN"}), None)
    cols = list(df.columns)
    target = st.sidebar.selectbox("Target column", cols, index=cols.index(guess) if guess else 0)

available = [m for m in MODEL_FILES if m in models]
if not available:
    st.error("No models could be loaded from `saved_models/`. Check the Model Status tab.")
selected = st.sidebar.selectbox("Model selector", available, format_func=lambda m: PRETTY.get(m, m)) if available else None



# ----------------------------------------------------------------- helpers
def get_feature_names(model):
    for attr in ("feature_names_in_", "feature_names_"):
        if hasattr(model, attr):
            try:
                return list(getattr(model, attr))
            except Exception:  # noqa: BLE001
                pass
    return None


def base_features():
    """Feature list used by the base models."""
    for name in ("random_forest", "xgboost", "lightgbm", "catboost", "stacking", "linear_regression"):
        if name in models:
            f = get_feature_names(models[name])
            if f:
                return f
    if df is not None and target is not None:
        return [c for c in df.select_dtypes("number").columns if c != target]
    return []


FEATURES = base_features()


def predict(name, model, X):
    """Predict with a few fallbacks; reports the real first error if all fail."""
    X = X[FEATURES] if FEATURES and all(f in X.columns for f in FEATURES) else X
    first_err = None
    try:
        return np.asarray(model.predict(X)).ravel()
    except Exception as e:  # noqa: BLE001
        first_err = e
    try:
        return np.asarray(model.predict(X.values)).ravel()
    except Exception:  # noqa: BLE001
        pass
    rf = models.get("random_forest")
    if rf is not None and name.startswith("hybrid"):
        try:
            extra = np.asarray(rf.predict(X)).reshape(-1, 1)
            return np.asarray(model.predict(np.hstack([X.values, extra]))).ravel()
        except Exception:  # noqa: BLE001
            pass
    raise ValueError(f"Could not predict with '{name}': {first_err}")


@st.cache_data
def get_test_data(_df, target_col, features):
    """Prefer saved_models/test_data.pkl (dict with X_test, y_test); otherwise re-split the CSV."""
    p = MODEL_DIR / "test_data.pkl"
    if p.exists():
        d = joblib.load(p)
        return pd.DataFrame(d["X_test"]), pd.Series(d["y_test"]).reset_index(drop=True), "saved"
    if _df is None or target_col is None or not features:
        return None, None, None
    data = _df[features + [target_col]].dropna()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data[features], data[target_col], test_size=0.2, random_state=RANDOM_STATE
    )
    return X_te, y_te.reset_index(drop=True), "resplit"


def normalise_comparison(c):
    if isinstance(c, dict):
        c = pd.DataFrame(c).T
    if not isinstance(c, pd.DataFrame):
        return None
    c = c.copy()
    if "Model" not in c.columns:
        c = c.reset_index()
        c = c.rename(columns={c.columns[0]: "Model"})
    return c


def find_col(frame, names):
    lower = {c.lower(): c for c in frame.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

def get_importance(model):
    """Return (names, values, label) or None."""
    pre, est, note = None, model, ""
    if hasattr(model, "named_steps"):
        pre = model.named_steps.get("preprocessor")
        est = list(model.named_steps.values())[-1]
    elif isinstance(model, XGBNative):
        pre, est = model.pre, model.model
    elif isinstance(model, StackingParts):
        pre, est, note = model.pre, model.rf, " (Random Forest inside stacking)"

    if hasattr(est, "feature_importances_"):
        vals, label = np.asarray(est.feature_importances_), "Built-in importance" + note
    elif hasattr(est, "coef_"):
        vals, label = np.abs(np.ravel(est.coef_)), "Absolute coefficient"
    else:
        return None

    names = list(pre.get_feature_names_out()) if pre is not None else FEATURES
    names = [n.split("__", 1)[-1] for n in names]
    if len(names) != len(vals):
        return None
    return names, vals, label

# ----------------------------------------------------------------- header
st.title("🩸 Sri Lankan Full Blood Count Dashboard")
st.subheader("Hemoglobin (HB) prediction using Bagging, Boosting, Stacking and Hybrid models")

tab_over, tab_cmp, tab_pred, tab_imp, tab_form, tab_status = st.tabs(
    ["📊 Dataset", "🏆 Model comparison", "📈 Actual vs predicted", "🔍 Feature importance",
     "🧪 Make a prediction", "⚙️ Model status"]
)

# ----------------------------------------------------------------- 1. dataset overview
with tab_over:
    if df is None:
        st.warning("No CSV found. Put your dataset in a `data/` folder next to app.py, or upload it in the sidebar.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing values", int(df.isna().sum().sum()))
        st.markdown("**Sample rows**")
        st.dataframe(df.head(20), width="stretch")
        st.markdown("**Summary statistics**")
        st.dataframe(df.describe().T, width="stretch")

        num = df.select_dtypes("number")
        if target in num.columns:
            a, b = st.columns(2)
            a.plotly_chart(px.histogram(df, x=target, nbins=40, title=f"Distribution of {target}"), width="stretch")
            corr = num.corr(numeric_only=True)
            b.plotly_chart(
                px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                          title="Correlation heatmap"),
                width="stretch",
            )

# ----------------------------------------------------------------- 2. model comparison
with tab_cmp:
    cpath = MODEL_DIR / "model_comparison.pkl"
    if not cpath.exists():
        st.warning("model_comparison.pkl was not found in saved_models/.")
    else:
        comp = normalise_comparison(load_pickle(str(cpath)))
        if comp is None:
            st.error("model_comparison.pkl is not a table (DataFrame/dict).")
        else:
            rmse_c = find_col(comp, ["RMSE"])
            r2_c = find_col(comp, ["R2", "R²", "R2 Score"])
            if rmse_c:
                best_idx = comp[rmse_c].astype(float).idxmin()
                crit = f"lowest {rmse_c}"
            elif r2_c:
                best_idx = comp[r2_c].astype(float).idxmax()
                crit = f"highest {r2_c}"
            else:
                best_idx, crit = None, ""

            if best_idx is not None:
                st.success(f"🏆 Best model: **{comp.loc[best_idx, 'Model']}** ({crit})")
                comp["Best"] = ["⭐" if i == best_idx else "" for i in comp.index]

            st.dataframe(comp, width="stretch", hide_index=True)

            metric_cols = [c for c in comp.columns if c not in ("Model", "Best")
                           and pd.api.types.is_numeric_dtype(comp[c])]
            if metric_cols:
                metric = st.selectbox("Metric to chart", metric_cols,
                                      index=metric_cols.index(rmse_c) if rmse_c in metric_cols else 0)
                colors = ["#2ca02c" if i == best_idx else "#4c78a8" for i in comp.index]
                fig = go.Figure(go.Bar(x=comp["Model"], y=comp[metric], marker_color=colors,
                                       text=comp[metric].round(3), textposition="outside"))
                fig.update_layout(title=f"{metric} by model (green = best)", xaxis_tickangle=-30)
                st.plotly_chart(fig, width="stretch")
                st.caption("For MAE, RMSE, MAPE and training time lower is better. For R² higher is better.")

# ----------------------------------------------------------------- 3. actual vs predicted
with tab_pred:
    if selected is None:
        st.info("No model available.")
    else:
        X_te, y_te, src = get_test_data(df, target, FEATURES)
        if X_te is None:
            st.warning("Need test data. Save `test_data.pkl` from the notebook (see instructions) or load the CSV.")
        else:
            if src == "resplit":
                st.caption("⚠️ Test set was re-created from the CSV (80/20 split). For exact notebook results, "
                           "save `test_data.pkl` from the notebook.")
            try:
                y_hat = predict(selected, models[selected], X_te)
                mae = mean_absolute_error(y_te, y_hat)
                rmse = float(np.sqrt(mean_squared_error(y_te, y_hat)))
                r2 = r2_score(y_te, y_hat)
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"{mae:.3f}")
                m2.metric("RMSE", f"{rmse:.3f}")
                m3.metric("R²", f"{r2:.3f}")

                lo, hi = float(min(y_te.min(), y_hat.min())), float(max(y_te.max(), y_hat.max()))
                fig = px.scatter(x=y_te, y=y_hat, labels={"x": f"Actual {target}", "y": f"Predicted {target}"},
                                 title=f"Actual vs predicted - {PRETTY.get(selected, selected)}", opacity=0.6)
                fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                                         line=dict(dash="dash", color="red")))
                st.plotly_chart(fig, width="stretch")

                resid = np.asarray(y_te) - y_hat
                st.plotly_chart(px.histogram(x=resid, nbins=40, labels={"x": "Residual (actual - predicted)"},
                                             title="Residual distribution"), width="stretch")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))

# ----------------------------------------------------------------- 4. feature importance
with tab_imp:
    if selected is None:
        st.info("No model available.")
    else:
        model = models[selected]
        info = get_importance(model)
        if info is not None:
            names, vals, label = info
            imp_df = pd.DataFrame({"Feature": names, "Importance": vals}).sort_values("Importance").tail(20)
            st.plotly_chart(px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                                   title=f"{label} - {PRETTY.get(selected, selected)}"), width="stretch")
        else:
            st.info("This model has no built-in importance (ANN, bagging, RF + ANN hybrid). "
                    "Use permutation importance below.")

        st.markdown("**Permutation importance** (works for any model; may take a few seconds)")
        if st.button("Compute permutation importance"):
            X_te, y_te, _ = get_test_data(df, target, FEATURES)
            if X_te is None:
                st.warning("No test data available.")
            else:
                try:
                    class _Wrap:
                        def __init__(self, n, m): self.n, self.m = n, m
                        def fit(self, *a, **k): return self
                        def predict(self, X): return predict(self.n, self.m, pd.DataFrame(X, columns=X_te.columns))
                        def score(self, X, y): return r2_score(y, self.predict(X))
                    res = permutation_importance(_Wrap(selected, model), X_te, y_te, n_repeats=10,
                                                 random_state=RANDOM_STATE, scoring=None)
                    p_df = pd.DataFrame({"Feature": X_te.columns, "Importance": res.importances_mean}
                                        ).sort_values("Importance")
                    st.plotly_chart(px.bar(p_df, x="Importance", y="Feature", orientation="h",
                                           title="Permutation importance (drop in R² when shuffled)"),
                                    width="stretch")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not compute: {e}")

# ----------------------------------------------------------------- 5. prediction form
with tab_form:
    st.markdown(f"Enter the blood-count values to predict **{target or 'HB'}**.")
    if not FEATURES:
        st.warning("Could not determine the input features. Check the Model Status tab.")
    else:
        values = {}
        cols = st.columns(3)
        for i, f in enumerate(FEATURES):
            col = cols[i % 3]
            if df is not None and f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
                s = df[f].dropna()
                lo, hi, med = float(s.min()), float(s.max()), float(s.median())
                values[f] = col.number_input(f, min_value=lo, max_value=hi, value=med,
                                             step=max((hi - lo) / 200, 0.01), format="%.2f")
            elif df is not None and f in df.columns:
                values[f] = col.selectbox(f, sorted(df[f].dropna().unique()))
            else:
                values[f] = col.number_input(f, value=0.0)

        if st.button("Predict", type="primary"):
            row = pd.DataFrame([values])
            results = {}
            for name in available:
                try:
                    results[PRETTY.get(name, name)] = float(predict(name, models[name], row)[0])
                except Exception:  # noqa: BLE001
                    results[PRETTY.get(name, name)] = np.nan
            res_df = pd.DataFrame({"Model": results.keys(), "Predicted": results.values()}).dropna()
            if res_df.empty:
                st.error("No model could make a prediction. Check the Model Status tab.")
            else:
                if selected:
                    st.metric(f"Selected model: {PRETTY.get(selected, selected)}",
                              f"{results.get(PRETTY.get(selected, selected), float('nan')):.2f}")
                st.plotly_chart(px.bar(res_df, x="Model", y="Predicted", text=res_df["Predicted"].round(2),
                                       title="Prediction from every model"), width="stretch")
                st.caption("Educational demo only. Always rely on a doctor / lab report for real decisions.")

# ----------------------------------------------------------------- 6. model status
with tab_status:
    c1, c2 = st.columns(2)
    c1.metric("Successfully loaded models", len(models))
    c2.metric("Models with errors", len(load_errors))
    for name, model in models.items():
        with st.expander(f"✅ {name}"):
            st.write("Model type:", type(model).__name__)
            if hasattr(model, "n_features_in_"):
                st.write("Number of input features:", model.n_features_in_)
            f = get_feature_names(model)
            if f:
                st.write("Input feature names:", f)
    for name, err in load_errors.items():
        st.error(f"{name}: {err}")
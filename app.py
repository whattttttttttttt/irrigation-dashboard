import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Irrigation Water Wastage - ML Dashboard", layout="wide")

MODELS = ["Random Forest", "XGBoost", "ANN", "SVR", "LSTM"]
COLORS = {"Random Forest": "#2ca02c", "XGBoost": "#ff7f0e", "ANN": "#9467bd",
          "SVR": "#d62728", "LSTM": "#1f77b4", "Observed": "#111111"}


@st.cache_data
def load():
    te = pd.read_csv("predictions_test.csv", parse_dates=["date"])
    tr = pd.read_csv("predictions_train.csv", parse_dates=["date"])
    return tr, te


train, test = load()


# ---------- metrics ----------
def metrics(o, p):
    o, p = np.asarray(o, float), np.asarray(p, float)
    ok = ~np.isnan(p)
    o, p = o[ok], p[ok]
    err = p - o
    r = np.corrcoef(o, p)[0, 1]
    nse = 1 - np.sum(err ** 2) / np.sum((o - o.mean()) ** 2)
    alpha, beta = p.std() / o.std(), p.mean() / o.mean()
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return {
        "R²": r ** 2, "NSE": nse, "KGE": kge,
        "RMSE": np.sqrt(np.mean(err ** 2)), "MAE": np.mean(np.abs(err)),
        "PBIAS (%)": 100 * np.sum(err) / np.sum(o),
        "RSR": np.sqrt(np.sum(err ** 2)) / np.sqrt(np.sum((o - o.mean()) ** 2)),
        "Correlation (r)": r,
    }


def metrics_table(df):
    rows = {m: metrics(df["observed"], df[m]) for m in MODELS}
    return pd.DataFrame(rows).T.round(3)


# ---------- plots ----------
def hydrograph(df, models, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["observed"], name="Observed",
                             line=dict(color=COLORS["Observed"], width=2)))
    for m in models:
        fig.add_trace(go.Scatter(x=df["date"], y=df[m], name=m,
                                 line=dict(color=COLORS[m], width=1.3)))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Discharge (m³/s)",
                      hovermode="x unified", height=430, legend=dict(orientation="h", y=-0.25))
    return fig


def scatter(df, models, title):
    fig = go.Figure()
    mx = max(df["observed"].max(), *[df[m].max() for m in models])
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="1:1 line",
                             line=dict(color="gray", dash="dash")))
    for m in models:
        fig.add_trace(go.Scatter(x=df["observed"], y=df[m], mode="markers", name=m,
                                 marker=dict(color=COLORS[m], size=6, opacity=0.6)))
    fig.update_layout(title=title, xaxis_title="Observed discharge (m³/s)",
                      yaxis_title="Predicted discharge (m³/s)", height=430,
                      legend=dict(orientation="h", y=-0.25))
    return fig


def taylor(df):
    obs_sd = df["observed"].std()
    fig = go.Figure()
    t = np.linspace(0, np.pi / 2, 100)
    rmax = 1.5
    # std arcs
    for rad in (0.5, 1.0, 1.5):
        fig.add_trace(go.Scatter(x=rad * np.cos(t), y=rad * np.sin(t), mode="lines",
                                 line=dict(color="blue", dash="dot", width=1), showlegend=False, hoverinfo="skip"))
    # centred-RMS arcs around the reference point (1,0)
    for c in (0.5, 1.0):
        a = np.linspace(0, np.pi, 200)
        x, y = 1 + c * np.cos(a), c * np.sin(a)
        keep = (x ** 2 + y ** 2 <= rmax ** 2 + 1e-9) & (y >= 0) & (x >= 0)
        fig.add_trace(go.Scatter(x=x[keep], y=y[keep], mode="lines",
                                 line=dict(color="lightgray", width=1.5), showlegend=False, hoverinfo="skip"))
    # correlation rays + labels
    for r in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        th = np.arccos(r)
        fig.add_trace(go.Scatter(x=[0, rmax * np.cos(th)], y=[0, rmax * np.sin(th)], mode="lines",
                                 line=dict(color="black", dash="dot", width=0.8), showlegend=False, hoverinfo="skip"))
        fig.add_annotation(x=1.06 * rmax * np.cos(th), y=1.06 * rmax * np.sin(th), text=str(r),
                           showarrow=False, font=dict(size=11))
    fig.add_annotation(x=1.2, y=1.2, text="Correlation", showarrow=False, textangle=-45, font=dict(size=12))
    # reference (observed)
    fig.add_trace(go.Scatter(x=[1], y=[0], mode="markers", name="Observed (reference)",
                             marker=dict(color="white", size=11, line=dict(color="black", width=2))))
    for m in MODELS:
        r = np.corrcoef(df["observed"], df[m])[0, 1]
        s = df[m].std() / obs_sd
        fig.add_trace(go.Scatter(x=[s * r], y=[s * np.sqrt(max(0, 1 - r ** 2))], mode="markers", name=m,
                                 marker=dict(color=COLORS[m], size=13),
                                 hovertemplate=f"{m}<br>r = {r:.3f}<br>Norm. SD = {s:.3f}<extra></extra>"))
    fig.update_xaxes(range=[-0.02, 1.7], title="Standard deviation (normalised)", zeroline=False,
                     showgrid=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-0.02, 1.7], title="Standard deviation (normalised)", zeroline=False, showgrid=False)
    fig.update_layout(title="Taylor Diagram - Discharge (test period)", height=650,
                      legend=dict(orientation="h", y=-0.18), plot_bgcolor="white")
    return fig


# ---------- layout ----------
st.sidebar.title("💧 Irrigation Water Wastage")
st.sidebar.caption("ML-based discharge prediction (RF, XGBoost, ANN, SVR, LSTM)")
page = st.sidebar.radio("Section", ["1. Hydrograph & Scatter", "2. Model Performance", "3. Taylor Diagram"])
period = st.sidebar.radio("Data period", ["Test (unseen)", "Training"],
                          help="Test = last 25% of the record, never seen by the models during training.")
df = test if period.startswith("Test") else train
if period == "Training":
    df = df.dropna()
st.sidebar.markdown(f"**{df['date'].min():%d %b %Y} → {df['date'].max():%d %b %Y}**  \n{len(df)} daily records")

if page.startswith("1"):
    st.header("Hydrograph & Scatter Plot")
    st.subheader("All models combined")
    c1, c2 = st.columns(2)
    c1.plotly_chart(hydrograph(df, MODELS, "Hydrograph - all models"), width="stretch")
    c2.plotly_chart(scatter(df, MODELS, "Scatter plot - all models"), width="stretch")

    st.divider()
    st.subheader("Individual model")
    model = st.selectbox("Select model", MODELS)
    c3, c4 = st.columns(2)
    c3.plotly_chart(hydrograph(df, [model], f"Hydrograph - {model}"), width="stretch")
    c4.plotly_chart(scatter(df, [model], f"Scatter plot - {model}"), width="stretch")
    m = metrics(df["observed"], df[model])
    k = st.columns(4)
    k[0].metric("R²", f"{m['R²']:.3f}")
    k[1].metric("NSE", f"{m['NSE']:.3f}")
    k[2].metric("RMSE", f"{m['RMSE']:.2f}")
    k[3].metric("MAE", f"{m['MAE']:.2f}")

elif page.startswith("2"):
    st.header("Model Performance Parameters")
    tbl = metrics_table(df)
    best = tbl["NSE"].idxmax()
    st.success(f"Best model by NSE ({period.split()[0].lower()}): **{best}** (NSE = {tbl.loc[best, 'NSE']:.3f})")
    st.dataframe(tbl.style.format("{:.3f}").highlight_max(subset=["R²", "NSE", "KGE", "Correlation (r)"], color="#c8f7c5")
                 .highlight_min(subset=["RMSE", "MAE", "RSR"], color="#c8f7c5"), width="stretch")
    st.download_button("Download table (CSV)", tbl.to_csv().encode(), "model_performance.csv")
    st.markdown("""
**What each parameter means**

| Parameter | Ideal | Meaning |
|---|---|---|
| R² | 1 | Share of variance in observed discharge explained by the model |
| NSE (Nash-Sutcliffe) | 1 | Model skill vs. just using the mean; below 0 = worse than the mean |
| KGE (Kling-Gupta) | 1 | Combines correlation, variability and bias in one score |
| RMSE | 0 | Typical error size (same unit as discharge), penalises large misses |
| MAE | 0 | Average absolute error |
| PBIAS (%) | 0 | Average over- (+) or under-prediction (−) |
| RSR | 0 | RMSE divided by standard deviation of observations |
| Correlation (r) | 1 | Strength of linear agreement |
""")
    with st.expander("Data & method notes"):
        st.markdown("""
- **Target:** daily river discharge. **Inputs:** catchment rainfall index (current, lagged and accumulated), groundwater level, previous-day suspended sediment, and seasonality.
- **Rainfall:** the 2021-2025 gauge files contain unusually large daily values, so each station is scaled by its own mean wet-day value and averaged into a dimensionless index.
- **Groundwater:** readings stop in Jan 2024, so the last value is held constant afterwards.
- **Split:** chronological 75% train / 25% test. All models use the same inputs, split and scaling.
""")

else:
    st.header("Taylor Diagram")
    st.plotly_chart(taylor(df), width="stretch")
    obs_sd = df["observed"].std()
    rows = []
    for m in MODELS:
        r = np.corrcoef(df["observed"], df[m])[0, 1]
        s = df[m].std()
        crmsd = np.sqrt(np.mean(((df[m] - df[m].mean()) - (df["observed"] - df["observed"].mean())) ** 2))
        rows.append({"Model": m, "Correlation": r, "Std dev": s, "Norm. std dev": s / obs_sd, "Centred RMSD": crmsd})
    st.dataframe(pd.DataFrame(rows).set_index("Model").round(3), width="stretch")
    st.caption("Closer to the white 'Observed' point (1, 0) = better. Angle = correlation, distance from origin = standard deviation, "
               "distance from the Observed point = centred RMS difference.")

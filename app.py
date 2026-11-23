import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Irrigation Water Wastage - ML Dashboard", layout="wide")

MODELS = ["Random Forest", "XGBoost", "ANN", "SVR", "LSTM"]
COLORS = {"Random Forest": "#2ca02c", "XGBoost": "#ff7f0e", "ANN": "#9467bd",
          "SVR": "#d62728", "LSTM": "#1f77b4", "Observed": "#7f7f7f"}
DATASETS = {
    "Discharge": dict(key="discharge", unit="m³/s", name="discharge",
        note="Daily river discharge (Nov 2021 - Nov 2025). Inputs: rainfall index, groundwater, sediment and their lags, plus seasonality."),
    "Groundwater level": dict(key="gw", unit="m", name="groundwater level",
        note="Groundwater level is measured only 3-4 times a year, so the daily series is interpolated (Jan 2019 - Jan 2024). "
             "Inputs: discharge, sediment, 30/60-day lagged groundwater level and seasonality. Treat the results as indicative."),
    "Rainfall": dict(key="rain", unit="index", name="rainfall",
        note="Rainfall is a dimensionless catchment index (each gauge scaled by its own mean wet-day value) because the raw gauge units were inconsistent. "
             "Rainfall is hard to predict from river and groundwater data, so skill is low."),
    "Suspended sediment": dict(key="sed", unit="source units", name="suspended sediment",
        note="Suspended sediment is near zero for most days of 2021-2025, so models have little signal to learn from. Treat the results as indicative."),
}


@st.cache_data
def load(key):
    return (pd.read_csv(f"pred_{key}_train.csv", parse_dates=["date"]),
            pd.read_csv(f"pred_{key}_test.csv", parse_dates=["date"]))


def corr(a, b):
    r = pd.Series(np.asarray(a, float)).corr(pd.Series(np.asarray(b, float)))
    return 0.0 if pd.isna(r) else float(r)


def metrics(o, p):
    o, p = np.asarray(o, float), np.asarray(p, float)
    ok = ~np.isnan(p); o, p = o[ok], p[ok]
    err = p - o
    r = corr(o, p)
    den = np.sum((o - o.mean()) ** 2) or 1e-12
    nse = 1 - np.sum(err ** 2) / den
    a = p.std() / (o.std() or 1e-12); b = p.mean() / (o.mean() or 1e-12)
    return {"R²": r ** 2, "NSE": nse, "KGE": 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2),
            "RMSE": np.sqrt(np.mean(err ** 2)), "MAE": np.mean(np.abs(err)),
            "PBIAS (%)": 100 * np.sum(err) / (np.sum(o) or 1e-12), "RSR": np.sqrt(np.sum(err ** 2) / den),
            "Correlation (r)": r}


def metrics_table(df):
    return pd.DataFrame({m: metrics(df["observed"], df[m]) for m in MODELS}).T


def hydrograph(df, models, title, unit):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["observed"], name="Observed", line=dict(color=COLORS["Observed"], width=2)))
    for m in models:
        fig.add_trace(go.Scatter(x=df["date"], y=df[m], name=m, line=dict(color=COLORS[m], width=1.3)))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=f"Value ({unit})", hovermode="x unified",
                      height=420, legend=dict(orientation="h", y=-0.25))
    return fig


def scatter(df, models, title, unit):
    fig = go.Figure()
    mx = max(df["observed"].max(), *[df[m].max() for m in models])
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="1:1 line", line=dict(color="gray", dash="dash")))
    for m in models:
        fig.add_trace(go.Scatter(x=df["observed"], y=df[m], mode="markers", name=m,
                                 marker=dict(color=COLORS[m], size=6, opacity=0.6)))
    fig.update_layout(title=title, xaxis_title=f"Observed ({unit})", yaxis_title=f"Predicted ({unit})",
                      height=420, legend=dict(orientation="h", y=-0.25))
    return fig


def taylor(df, label):
    GOLD, ROSE, GREY = "#d4a017", "#d35400", "#7f8c8d"
    obs = df["observed"]; osd = obs.std() or 1e-12
    pts = {m: (corr(obs, df[m]), df[m].std() / osd) for m in MODELS}
    rmax = max(1.5, np.ceil(max(s for _, s in pts.values()) * 2 + 0.2) / 2)
    t = np.linspace(0, np.pi / 2, 120)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rmax * np.cos(t), y=rmax * np.sin(t), mode="lines", line=dict(color=GREY, width=1.5),
                             showlegend=False, hoverinfo="skip"))
    for rad in np.arange(0.5, rmax, 0.5):
        fig.add_trace(go.Scatter(x=rad * np.cos(t), y=rad * np.sin(t), mode="lines",
                                 line=dict(color="#a9cce3", dash="dash", width=1), showlegend=False, hoverinfo="skip"))
    for c in (0.5, 1.0, 1.5):
        a = np.linspace(0, np.pi, 240); x, y = 1 + c * np.cos(a), c * np.sin(a)
        k = (x ** 2 + y ** 2 <= rmax ** 2) & (y >= 0) & (x >= 0)
        fig.add_trace(go.Scatter(x=x[k], y=y[k], mode="lines", line=dict(color=GOLD, dash="dash", width=2),
                                 showlegend=False, hoverinfo="skip"))
        ang = np.pi * 0.72; lx, ly = 1 + c * np.cos(ang), c * np.sin(ang)
        if lx > 0.03 and lx ** 2 + ly ** 2 < rmax ** 2:
            fig.add_annotation(x=lx, y=ly, text=f"{c}", showarrow=False, font=dict(color=GOLD, size=13))
    for r in (0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99):
        th = np.arccos(r)
        fig.add_trace(go.Scatter(x=[0, rmax * np.cos(th)], y=[0, rmax * np.sin(th)], mode="lines",
                                 line=dict(color="#aab7b8", dash="dot", width=0.9), showlegend=False, hoverinfo="skip"))
        fig.add_annotation(x=1.05 * rmax * np.cos(th), y=1.05 * rmax * np.sin(th), text=str(r), showarrow=False,
                           font=dict(color=ROSE, size=13))
    fig.add_annotation(x=0.93 * rmax * np.cos(0.62), y=0.93 * rmax * np.sin(0.62) + 0.14 * rmax, text="Correlation", showarrow=False,
                       textangle=-35, font=dict(color=ROSE, size=13))
    fig.add_trace(go.Scatter(x=[1], y=[0], mode="markers", name="Reference (observed)",
                             marker=dict(symbol="star", color="#1b2631", size=13, line=dict(color=GREY, width=1))))
    for m, (r, s) in pts.items():
        fig.add_trace(go.Scatter(x=[s * r], y=[s * np.sqrt(max(0, 1 - r ** 2))], mode="markers", name=m,
                                 marker=dict(color=COLORS[m], size=12, line=dict(color="#1b2631", width=1.5)),
                                 hovertemplate=f"{m}<br>r = {r:.3f}<br>Norm. SD = {s:.3f}<extra></extra>"))
    ax = dict(range=[-0.02, rmax + 0.2], zeroline=False, showgrid=False, tickfont=dict(color=GREY, size=12),
              title_font=dict(color=GREY), linecolor=GREY)
    fig.update_xaxes(title="Standard deviation (normalised)", scaleanchor="y", scaleratio=1, **ax)
    fig.update_yaxes(title="Standard deviation (normalised)", **ax)
    fig.update_layout(title=dict(text=f"Taylor Diagram - {label}", font=dict(color=GREY, size=15)), height=470,
                      margin=dict(l=40, r=20, t=50, b=40), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top", bgcolor="rgba(0,0,0,0)", font=dict(color=GREY)))
    return fig, pts


def hi_mask(obs):
    pos = obs[obs > 0]
    thr = pos.quantile(0.9) if len(pos) > 10 else obs.quantile(0.9)
    return obs >= thr, thr


def conclusions(df, tbl, cfg):
    best, worst = tbl["NSE"].idxmax(), tbl["NSE"].idxmin()
    mask, thr = hi_mask(df["observed"]); hi = df[mask]
    pb = {m: 100 * (hi[m] - hi["observed"]).sum() / (hi["observed"].sum() or 1e-12) for m in MODELS}
    under = sum(v < 0 for v in pb.values())
    out = [f"**{best}** tracks the observed {cfg['name']} best (NSE {tbl.loc[best,'NSE']:.2f}, R² {tbl.loc[best,'R²']:.2f}); "
           f"**{worst}** is the weakest (NSE {tbl.loc[worst,'NSE']:.2f})."]
    if tbl.loc[best, "NSE"] < 0:
        out.append("No model beats simply using the mean (NSE below 0), so these predictions are only indicative.")
    elif tbl.loc[best, "NSE"] < 0.5:
        out.append("Skill is modest (NSE below 0.5), so the models capture the general pattern but not the detail.")
    else:
        out.append("Skill is good (NSE above 0.5): the models reproduce the timing and shape of the observed series.")
    out.append(f"{under} of 5 models under-predict the highest values (top 10% days), so peaks are generally {'smoothed out' if under >= 3 else 'captured reasonably'}.")
    out.append(cfg["note"])
    return out


def interpretation(df, tbl, cfg):
    obs, u, nm = df["observed"], cfg["unit"], cfg["name"]
    best = tbl["NSE"].idxmax()
    pk = df.loc[obs.idxmax()]
    mon = df.groupby(df["date"].dt.month)["observed"].mean()
    names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    top = [names[i] for i in mon.sort_values(ascending=False).index[:3]]
    low = [names[i] for i in mon.sort_values().index[:2]]
    err = (pk[best] - pk["observed"]) / (abs(pk["observed"]) or 1e-12) * 100
    lag = max(range(-5, 6), key=lambda k: corr(obs.values[max(0, k):len(obs) + min(0, k)], df[best].values[max(0, -k):len(obs) - max(0, k)]))
    slope = np.polyfit(obs, df[best], 1)[0] if obs.std() > 0 else np.nan
    read = [
        f"**Hydrograph (line graph):** x-axis is time, y-axis is the {nm} ({u}). The grey line is what was actually observed; each coloured line is a model's prediction. The closer a coloured line follows the grey line, the better the model.",
        "**Peaks and dips:** a peak is a sharp rise in the line (a wet spell, flood or high reading); flat low stretches are dry periods. Models that miss the top of a peak are under-predicting extremes.",
        "**Scatter plot:** each dot is one day, observed value on the x-axis and predicted value on the y-axis. Dots on the dashed 1:1 line are perfect predictions, dots below it are under-predictions, dots above it are over-predictions. A tight cluster along the line means a good model; a wide cloud means a poor one.",
    ]
    see = [
        f"**Peak:** the highest observed {nm} was **{pk['observed']:.4g} {u}** on **{pk['date']:%d %b %Y}**. On that day the best model ({best}) predicted {pk[best]:.4g} {u}, which is {abs(err):.0f}% {'below' if err < 0 else 'above'} the observed value.",
        f"**Seasonality:** the highest average values fall in **{', '.join(top)}**, and the lowest in **{', '.join(low)}**, showing the seasonal rhythm of the series.",
        f"**Timing:** the {best} line {'lines up with the observed line on the same day' if lag == 0 else ('follows the observed line with a delay of about ' + str(abs(lag)) + ' day(s)' if lag > 0 else 'moves about ' + str(abs(lag)) + ' day(s) ahead of the observed line')}.",
    ]
    if not np.isnan(slope):
        see.append(f"**Scatter fit:** the {best} dots have a trend slope of {slope:.2f} against the 1:1 line (1.00 is perfect). "
                   + ("A slope below 1 means the model squeezes the range: highs are predicted too low and lows too high." if slope < 0.9 else
                      "A slope above 1 means the model exaggerates the range." if slope > 1.1 else "The model reproduces the spread of the data well."))
    good = tbl.loc[best, "NSE"] >= 0.5
    pred = [
        f"The models learn how {nm} responds to the other measured variables and the season, so for new days they can estimate {nm} from those inputs." if good else
        f"Predictive power for {nm} is limited here (best NSE {tbl.loc[best,'NSE']:.2f}); use it to see the broad trend and to compare models, not for exact day-to-day values.",
        f"**{best}** is the model to use for this dataset. The ranking of models also shows which method suits {nm} data, which is the main point of comparing five models.",
        "These results feed into the irrigation water wastage study: reliable discharge, rainfall and groundwater predictions show when water is plentiful or scarce, which helps judge when irrigation supply is wasted or short.",
    ]
    return read, see, pred


def qa(df, tbl, cfg):
    WHY = {"Random Forest": "it averages many decision trees, which is stable but smooths sharp peaks",
           "XGBoost": "it builds trees one after another, each correcting the last, capturing non-linear effects well",
           "ANN": "its hidden layers learn non-linear links between the input variables",
           "SVR": "it fits a smooth non-linear curve and is robust to noise",
           "LSTM": "it remembers the previous 14 days of inputs, which suits hydrological processes that depend on recent history"}
    Q = [f"Which model predicts {cfg['name']} best, and why?", "When was the largest observed value, and how well did models capture it?",
         "Do the models under-predict high values?", "Which model is closest to the observed data on the Taylor diagram?"]
    cols = st.columns(2)
    for i, q in enumerate(Q):
        if cols[i % 2].button(q, width="stretch", key=f"q{i}"):
            st.session_state["q"] = i
    qi = st.session_state.get("q")
    if qi is None:
        st.info("Click a question above."); return
    obs, u = df["observed"], cfg["unit"]
    st.markdown(f"**{Q[qi]}**")
    if qi == 0:
        best = tbl["NSE"].idxmax(); rk = tbl["NSE"].sort_values(ascending=False)
        st.success(f"**{best}**: NSE {tbl.loc[best,'NSE']:.3f}, R² {tbl.loc[best,'R²']:.3f}, RMSE {tbl.loc[best,'RMSE']:.3g}, MAE {tbl.loc[best,'MAE']:.3g}.")
        st.write(f"Why: {WHY[best]}.")
        st.write("Ranking by NSE: " + " > ".join(f"{m} ({v:.2f})" for m, v in rk.items()) + ".")
    elif qi == 1:
        d = df.loc[obs.idxmax()]
        dif = {m: abs(d[m] - d["observed"]) for m in MODELS}; cl = min(dif, key=dif.get)
        st.success(f"Largest observed value: **{d['observed']:.4g} {u}** on **{d['date']:%d %b %Y}**.")
        st.dataframe(pd.DataFrame({f"Predicted ({u})": {m: d[m] for m in MODELS}, "Error": {m: d[m] - d["observed"] for m in MODELS}}).round(4), width="stretch")
        st.write(f"Closest on that day: **{cl}**.")
    elif qi == 2:
        mask, thr = hi_mask(obs); hi = df[mask]
        pb = {m: 100 * (hi[m] - hi["observed"]).sum() / (hi["observed"].sum() or 1e-12) for m in MODELS}
        st.write(f"High values = days at or above the 90th percentile of observed values ({thr:.4g} {u}), {len(hi)} days.")
        st.dataframe(pd.DataFrame({"High-value PBIAS (%)": pb}).round(1), width="stretch")
        un = [m for m in MODELS if pb[m] < 0]
        st.success((f"Yes: {', '.join(un)} under-predict high values; worst is **{min(pb, key=pb.get)}** ({min(pb.values()):.1f}%). Negative PBIAS = predictions fall short.") if un else "No model under-predicts high values.")
    else:
        dist = {}
        for m in MODELS:
            r, s = corr(obs, df[m]), df[m].std() / (obs.std() or 1e-12)
            dist[m] = float(np.sqrt(max(0, 1 + s ** 2 - 2 * s * r)))
        b = min(dist, key=dist.get)
        st.success(f"**{b}** is closest to the observed (reference) point, distance {dist[b]:.3f}.")
        st.dataframe(pd.DataFrame({"Distance from reference": dist}).sort_values("Distance from reference").round(3), width="stretch")


# ---------------- layout ----------------
st.sidebar.title("💧 Irrigation Water Wastage")
st.sidebar.caption("ML-based prediction (RF, XGBoost, ANN, SVR, LSTM)")
page = st.sidebar.radio("Section", ["1. Hydrograph & Scatter", "2. Model Performance", "3. Taylor Diagram"])
period = st.sidebar.radio("Data period", ["Test (unseen)", "Training"], help="Test = last 25% of the record, never seen by the models during training.")

ds = st.selectbox("📊 Select dataset", list(DATASETS), key="ds")
cfg = DATASETS[ds]
train, test = load(cfg["key"])
df = test if period.startswith("Test") else train.dropna()
st.sidebar.markdown(f"**{df['date'].min():%d %b %Y} → {df['date'].max():%d %b %Y}**  \n{len(df)} daily records")
tbl = metrics_table(df)

if page.startswith("1"):
    st.header(f"Hydrograph & Scatter Plot - {ds}")
    st.subheader("All models combined")
    c1, c2 = st.columns(2)
    c1.plotly_chart(hydrograph(df, MODELS, f"{ds} hydrograph - all models", cfg["unit"]), width="stretch")
    c2.plotly_chart(scatter(df, MODELS, f"{ds} scatter plot - all models", cfg["unit"]), width="stretch")
    st.divider()
    st.subheader("Individual model")
    model = st.selectbox("Select model", MODELS)
    c3, c4 = st.columns(2)
    c3.plotly_chart(hydrograph(df, [model], f"{ds} hydrograph - {model}", cfg["unit"]), width="stretch")
    c4.plotly_chart(scatter(df, [model], f"{ds} scatter plot - {model}", cfg["unit"]), width="stretch")
    m = tbl.loc[model]; k = st.columns(4)
    k[0].metric("R²", f"{m['R²']:.3f}"); k[1].metric("NSE", f"{m['NSE']:.3f}")
    k[2].metric("RMSE", f"{m['RMSE']:.3g}"); k[3].metric("MAE", f"{m['MAE']:.3g}")
    st.divider()
    st.subheader("Conclusions")
    for c in conclusions(df, tbl, cfg):
        st.markdown(f"- {c}")
    rd, se, pr = interpretation(df, tbl, cfg)
    with st.expander("📖 How to read these graphs", expanded=True):
        for c in rd: st.markdown(f"- {c}")
    with st.expander("🔍 What the graphs show for this dataset", expanded=True):
        for c in se: st.markdown(f"- {c}")
    with st.expander("🔮 What can be predicted / how to use this", expanded=True):
        for c in pr: st.markdown(f"- {c}")
    st.divider()
    st.subheader("Ask the Dashboard")
    st.caption("Pick a question. Answers are calculated live from the model results for the selected dataset and period.")
    qa(df, tbl, cfg)

elif page.startswith("2"):
    st.header(f"Model Performance Parameters - {ds}")
    best = tbl["NSE"].idxmax()
    st.success(f"Best model by NSE: **{best}** (NSE = {tbl.loc[best,'NSE']:.3f})")
    st.dataframe(tbl.style.format("{:.3f}").highlight_max(subset=["R²", "NSE", "KGE", "Correlation (r)"], color="#c8f7c5")
                 .highlight_min(subset=["RMSE", "MAE", "RSR"], color="#c8f7c5"), width="stretch")
    st.download_button("Download table (CSV)", tbl.round(4).to_csv().encode(), f"performance_{cfg['key']}.csv")
    st.markdown("""
| Parameter | Ideal | Meaning |
|---|---|---|
| R² | 1 | Share of variance in observed values explained by the model |
| NSE (Nash-Sutcliffe) | 1 | Skill vs. just using the mean; below 0 = worse than the mean |
| KGE (Kling-Gupta) | 1 | Combines correlation, variability and bias |
| RMSE | 0 | Typical error size (same unit as the data), penalises large misses |
| MAE | 0 | Average absolute error |
| PBIAS (%) | 0 | Average over- (+) or under-prediction (−) |
| RSR | 0 | RMSE divided by the standard deviation of observations |
| Correlation (r) | 1 | Strength of linear agreement |
""")
    with st.expander("Data & method notes"):
        st.markdown(f"- {cfg['note']}\n- Chronological 75% train / 25% test split; every model uses the same inputs, split and scaling.")

else:
    st.header(f"Taylor Diagram - {ds}")
    fig, pts = taylor(df, ds)
    _, mid, _ = st.columns([1, 2.2, 1])
    mid.plotly_chart(fig, width="stretch")
    rows = []
    for m in MODELS:
        r, s = pts[m]
        rows.append({"Model": m, "Correlation": r, "Norm. std dev": s, "Centred RMSD": float(np.sqrt(max(0, 1 + s ** 2 - 2 * s * r)))})
    st.dataframe(pd.DataFrame(rows).set_index("Model").round(3), width="stretch")
    st.caption("Angle = correlation, distance from origin = normalised standard deviation, distance from the star (observed) = centred RMS difference. Closer to the star = better.")

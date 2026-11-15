from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AQI forecasting, Manali (Chennai)", layout="wide")

DATA = Path(__file__).resolve().parent / "data" / "processed"
MODELS = {
    "persistence": "Persistence (tomorrow = today)",
    "arima": "ARIMA(2,0,0)",
    "lstm_aqi_only": "LSTM (AQI only)",
    "lstm_with_pollutants": "LSTM (AQI + pollutants)",
}
EXPECTED_MAE = {"persistence": 24.38, "arima": 25.71, "lstm_aqi_only": 23.33, "lstm_with_pollutants": 26.35}


def category(x):
    if x <= 50: return "Good"
    if x <= 100: return "Satisfactory"
    if x <= 200: return "Moderate"
    if x <= 300: return "Poor"
    if x <= 400: return "Very Poor"
    return "Severe"


@st.cache_data
def load():
    series = pd.read_csv(DATA / "manali_model.csv", index_col="Date", parse_dates=True)
    preds = pd.read_csv(DATA / "preds_test_all.csv", index_col=0, parse_dates=True)
    return series, preds


def scores(actual, pred):
    err = actual - pred
    return {"MAE": err.abs().mean(),
            "RMSE": float(np.sqrt((err ** 2).mean())),
            "sMAPE %": float((200 * err.abs() / (actual.abs() + pred.abs())).mean())}


series, preds = load()
first, last = preds.index.min().date(), preds.index.max().date()

st.title("AQI forecasting: Manali, Chennai")
st.info("Replay of held-out history (23 Feb to 21 May 2019), not a live system. Every forecast for a day "
        "uses data up to the previous day only. Station: Manali (CPCB), an industrial-area station near the "
        "refinery cluster. No weather data was available.")

st.sidebar.header("Controls")
chosen = st.sidebar.multiselect("Models", list(MODELS), default=list(MODELS), format_func=MODELS.get)
d0, d1 = st.sidebar.slider("Test-period dates", min_value=first, max_value=last, value=(first, last))
event_thr = st.sidebar.slider("An event is an actual AQI above", 100, 300, 200, step=10)
trigger_thr = st.sidebar.slider("Raise an alert when the forecast is at least", 100, 300, 200, step=10)

with st.sidebar.expander("Self-check"):
    st.write("Recomputes the test-period MAE of each model and compares it with the notebook.")
    if st.button("Run check"):
        for k, expected in EXPECTED_MAE.items():
            got = scores(preds["actual"], preds[k])["MAE"]
            msg = f"{MODELS[k]}: {got:.2f} (expected {expected})"
            (st.success if abs(got - expected) < 0.01 else st.error)(msg)

if not chosen:
    st.warning("Select at least one model.")
    st.stop()
win = preds.loc[pd.Timestamp(d0):pd.Timestamp(d1)]
if win.empty:
    st.warning("No days in the selected range.")
    st.stop()

t_plot, t_score, t_alert, t_day, t_about = st.tabs(
    ["Forecasts", "Scorecard", "Alert simulator", "Day view", "About"])

with t_plot:
    show_hist = st.checkbox("Show the history before the test period", value=False)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if show_hist:
        ax.plot(series.index, series["AQI"], color="lightgrey", lw=0.8, label="history")
    ax.plot(win.index, win["actual"], color="black", lw=1.4, label="actual")
    for k in chosen:
        ax.plot(win.index, win[k], lw=1, label=MODELS[k])
    ax.axhline(event_thr, color="red", ls="--", lw=0.8, label=f"event level ({event_thr})")
    ax.set_ylabel("AQI")
    ax.legend(fontsize=8, ncol=3)
    st.pyplot(fig)
    plt.close(fig)
    st.write("The forecasts follow the actual line about a day late and underpredict the peaks: "
             "the models only know a spike after it has started.")

with t_score:
    tab = pd.DataFrame({MODELS[k]: scores(win["actual"], win[k]) for k in chosen}).T
    base = scores(win["actual"], win["persistence"])["MAE"]
    tab["skill vs persistence (MAE)"] = 1 - tab["MAE"] / base
    st.dataframe(tab.round(2))
    st.write("Skill is the share by which a model's average error is smaller than persistence "
             "(0 = no better, negative = worse). Over the full test period, bootstrap intervals show no "
             "model is significantly better than persistence on MAE; ARIMA and the LSTMs have a lower RMSE.")

with t_alert:
    ev = win["actual"] > event_thr
    st.write(f"{int(ev.sum())} event day(s) in the selected range. Very few events: treat as indicative.")
    rows = {}
    for k in chosen:
        al = win[k] >= trigger_thr
        rows[MODELS[k]] = {"caught": int((ev & al).sum()), "missed": int((ev & ~al).sum()),
                           "false alarms": int((~ev & al).sum())}
    st.dataframe(pd.DataFrame(rows).T)
    levels = np.arange(100, 301, 10)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    for k in chosen:
        ax[0].plot(levels, [int((ev & (win[k] >= t)).sum()) for t in levels], label=MODELS[k])
        ax[1].plot(levels, [int((~ev & (win[k] >= t)).sum()) for t in levels], label=MODELS[k])
    ax[0].set_title("Events caught")
    ax[1].set_title("False alarms")
    for a in ax:
        a.set_xlabel("alert level on the forecast")
    ax[0].legend(fontsize=7)
    st.pyplot(fig)
    plt.close(fig)
    st.write("Lowering the alert level catches more events but raises more false alarms: "
             "the trade-off an operator has to choose.")

with t_day:
    day = st.selectbox("Day", list(win.index.date))
    row = win.loc[pd.Timestamp(day)]
    prev = series["AQI"].get(pd.Timestamp(day) - pd.Timedelta(days=1))
    if prev is not None:
        st.write(f"AQI on the previous day (what the models saw): {prev:.0f} ({category(prev)})")
    rows = [{"": "Actual", "AQI": round(row["actual"]), "Category": category(row["actual"]), "Alert": ""}]
    for k in chosen:
        rows.append({"": MODELS[k], "AQI": round(row[k]), "Category": category(row[k]),
                     "Alert": "ALERT" if row[k] >= trigger_thr else "no alert"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

with t_about:
    st.markdown("""
**Data.** CPCB daily data (Kaggle: Air Quality Data in India 2015-2020), station Manali, Chennai (CPCB).
Series used: 2017-10-15 to 2019-05-21 (584 days), the longest near-continuous stretch; short gaps (up to 5 days)
were interpolated. Split by date: 70% train, 15% validation, 15% test (88 days).

**Models.** Persistence (tomorrow = today); ARIMA(2,0,0) chosen on validation error, no differencing needed
(ADF and KPSS agree the series is stationary); LSTM in PyTorch with a 14-day window, averaged over 5 random seeds,
with and without lagged pollutants (PM2.5, NO2, SO2, NH3).

**Limits.** One station near, not at, the refinery fence line. No weather data, while spikes are largely
weather-driven. CO unusable (stuck sensor); PM10, O3 and hydrocarbons missing. Only a handful of event days in the
test period. Less than two years of data, so the annual cycle cannot be learned reliably. This is a replay of
held-out history, not a live forecasting system.
""")
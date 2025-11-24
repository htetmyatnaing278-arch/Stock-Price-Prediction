import os
import random
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objs as go
import tensorflow as tf
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta

# -----------------------------
# Reproducibility
# -----------------------------
random_seed = 123
random.seed(random_seed)
np.random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(random_seed)

# -----------------------------
# Load model, scaler, window size
# -----------------------------
@st.cache_resource
def load_saved_components():
    model_path = "aapl_lstm_streamlit_app/lstm_aapl_model.h5"
    scaler_path = "aapl_lstm_streamlit_app/scaler.pkl"
    window_path = "aapl_lstm_streamlit_app/window_size.txt"

    if not os.path.exists(model_path):
        st.error("Model missing")
        st.stop()

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    with open(window_path) as f:
        window_size = int(f.read().strip())

    return model, scaler, window_size


# -----------------------------
# Safe Close price fetcher
# -----------------------------
@st.cache_data
def fetch_close_list(ticker, start_date, num_needed):
    """
    Fetch close price list safely (handles empty df or missing Close column)
    """
    end_date = start_date + timedelta(days=num_needed * 3)

    df = yf.download(ticker, start=start_date, end=end_date)

    if df is None or df.empty:
        return []

    # FIXED: Ensure Close column exists
    close_col = None
    for c in df.columns:
        if "close" in c.lower():
            close_col = c
            break

    if close_col is None:
        return []

    closes = df[close_col].dropna().tolist()

    if len(closes) < num_needed:
        return closes  # partial data

    return closes[:num_needed]


# -----------------------------
# Prediction Function
# -----------------------------
def predict_next_days(model, scaler, recent_values, days, window_size):
    scaled = scaler.transform(np.array(recent_values).reshape(-1, 1))
    scaled_list = list(scaled.flatten())

    preds_scaled = []
    for _ in range(days):
        seq = np.array(scaled_list[-window_size:]).reshape(1, window_size, 1)
        p = model.predict(seq, verbose=0)
        preds_scaled.append(p[0][0])
        scaled_list.append(p[0][0])

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AAPL Predictor", layout="wide")
st.title("AAPL LSTM Stock Close Price Predictor")

model, scaler, window_size = load_saved_components()
st.success(f"Model Loaded — window_size = {window_size}")

# Manual Input Section
st.subheader("Manual Input (Auto-filled from 2025-01-01)")

START_DATE = datetime(2025, 1, 1)
TICKER = "AAPL"

# Fetch historical values
history_list = fetch_close_list(TICKER, START_DATE, window_size)

if len(history_list) < window_size:
    st.warning(f"Could not fetch enough closes. Filling missing values.")
    while len(history_list) < window_size:
        history_list.append(round(random.uniform(160, 180), 2))

# Default text for manual box
default_text = ",".join([str(x) for x in history_list])

manual_text = st.text_area(
    f"Enter last {window_size} closing prices:",
    value=default_text,
    height=100
)

days = st.number_input("Days to predict", min_value=1, max_value=30, value=7)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict"):
    try:
        values = [float(v.strip()) for v in manual_text.split(",")]

        if len(values) < window_size:
            st.error("Not enough values.")
            st.stop()

        values = values[-window_size:]  # ensure correct window

        preds = predict_next_days(model, scaler, values, days, window_size)

        # Create Dates
        history_dates = [START_DATE + timedelta(days=i) for i in range(window_size)]
        pred_dates = [history_dates[-1] + timedelta(days=i + 1) for i in range(days)]

        # Combined for connecting line
        pred_x = [history_dates[-1]] + pred_dates
        pred_y = [values[-1]] + list(preds)

        # -----------------------------
        # Plot (TRAIN/TEST/PRED STYLE)
        # -----------------------------
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=history_dates,
            y=values,
            mode="lines",
            name="History",
            line=dict(color="black", width=2)
        ))

        fig.add_trace(go.Scatter(
            x=pred_x,
            y=pred_y,
            mode="lines+markers",
            name="Predicted",
            line=dict(color="blue", width=2)
        ))

        fig.update_layout(
            title="AAPL Close Price — Manual Input Prediction",
            xaxis_title="Date",
            yaxis_title="Close Price ($)",
            plot_bgcolor="lightyellow",
            paper_bgcolor="lightyellow"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show prediction table
        st.write("### Predicted Values")
        df_pred = pd.DataFrame({"Predicted Close": preds}, index=pred_dates)
        st.dataframe(df_pred)

    except Exception as e:
        st.error(f"Error: {e}")


st.markdown("---")

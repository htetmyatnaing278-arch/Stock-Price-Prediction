# app.py
import os
import random
from datetime import datetime, timedelta
import math

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objs as go
import tensorflow as tf
from tensorflow.keras.models import load_model
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# Reproducibility
# -----------------------------
random_seed = 123
random.seed(random_seed)
np.random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
try:
    tf.random.set_seed(random_seed)
except Exception:
    pass

# -----------------------------
# Constants / Defaults
# -----------------------------
MODEL_PATH = 'aapl_lstm_streamlit_app/lstm_aapl_model.h5'
SCALER_PATH = 'aapl_lstm_streamlit_app/scaler.pkl'
WINDOW_PATH = 'aapl_lstm_streamlit_app/window_size.txt'
MANUAL_DEFAULT_START = datetime(2025, 1, 1)
HISTORICAL_START = datetime(2020, 1, 1)  # for train/test example
HISTORICAL_TICKER = "AAPL"

# -----------------------------
# Helpers: load saved components
# -----------------------------
@st.cache_resource
def load_saved_components(model_path=MODEL_PATH, scaler_path=SCALER_PATH, window_path=WINDOW_PATH):
    # Validate existence
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")
    if not os.path.exists(window_path):
        raise FileNotFoundError(f"Window size file not found at: {window_path}")

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    with open(window_path, 'r') as f:
        window_size = int(f.read().strip())

    return model, scaler, window_size

# -----------------------------
# Helpers: fetch data
# -----------------------------
@st.cache_data
def fetch_close_dataframe(ticker, start, end):
    """Return DataFrame with DatetimeIndex and 'Close' column."""
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        return pd.DataFrame()
    df = df[['Close']].dropna()
    return df

@st.cache_data
def fetch_close_list(ticker, start_date, num_days):
    """Return list of close prices (num_days trading days) starting from start_date."""
    # Use an end date buffer to capture enough trading days
    end_date = start_date + timedelta(days=num_days * 3)
    df = fetch_close_dataframe(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    closes = df['Close'].dropna().tolist()
    return closes[:num_days]

# -----------------------------
# Helpers: prediction utilities
# -----------------------------
def predict_next_days(model, scaler, recent_values, days, window_size):
    """
    recent_values: 1D array-like of the most recent raw-close values (length >= window_size recommended)
    returns: array of predicted raw-close values (length = days)
    """
    arr = np.array(recent_values).astype(float).reshape(-1, 1)
    scaled = scaler.transform(arr)
    scaled_list = list(scaled.flatten())

    preds_scaled = []
    for _ in range(days):
        seq = np.array(scaled_list[-window_size:]).reshape(1, window_size, 1)
        p = model.predict(seq, verbose=0)
        pred_val = float(p.flatten()[0])
        preds_scaled.append(pred_val)
        scaled_list.append(pred_val)

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    return preds

def sliding_window_predictions_on_series(model, scaler, series, window_size):
    """
    Produces predictions for positions after the training portion using a rolling window.
    series: 1D numpy array of raw Close prices
    returns: predicted array aligned to the windows (len = len(series)-window_size)
    """
    arr = np.array(series).reshape(-1, 1)
    scaled = scaler.transform(arr)
    preds = []
    for i in range(len(scaled) - window_size):
        seq = scaled[i:i + window_size].reshape(1, window_size, 1)
        p = model.predict(seq, verbose=0)
        preds.append(float(p.flatten()[0]))
    preds = np.array(preds).reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds).flatten()
    return preds_inv  # corresponds to predictions for windows starting at 0 -> predicts value at index window_size, window_size+1, ...

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# -----------------------------
# Streamlit UI / Layout
# -----------------------------
st.set_page_config(page_title='AAPL — LSTM Predictor (Rewritten)', layout='wide')
st.title("AAPL Close Price — LSTM Predictor (Clean & Modular)")

# Load model / scaler / window_size and handle errors gracefully
try:
    model, scaler, window_size = load_saved_components()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Unexpected error loading model/scaler/window: {e}")
    st.stop()

st.success(f"Loaded model and scaler — window_size = {window_size}")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    days_to_predict = st.number_input("Days to predict (future)", min_value=1, max_value=30, value=7)
    manual_start = st.date_input("Manual input start date (manual box) — fixed to 2025-01-01", MANUAL_DEFAULT_START)
    use_auto_fill = st.checkbox("Auto-fill manual box using AAPL closes from 2025-01-01", value=True)
    show_metrics = st.checkbox("Show historical test metrics (MAE & RMSE)", value=True)
    # Historical section controls
    st.markdown("---")
    st.subheader("Historical comparison setup")
    hist_start = st.date_input("Historical data start (for train/test)", HISTORICAL_START)
    hist_end = st.date_input("Historical data end (for train/test)", datetime.now().date())
    hist_train_pct = st.slider("Train split (%)", min_value=50, max_value=90, value=80)

# -----------------------------
# Auto-fill manual input (starts at 2025-01-01)
# -----------------------------
# Always attempt to grab window_size closes from MANUAL_DEFAULT_START
manual_default_list = []
if use_auto_fill:
    manual_default_list = fetch_close_list(HISTORICAL_TICKER, MANUAL_DEFAULT_START, window_size)
    if len(manual_default_list) < window_size:
        # If not enough trading days found, pad with last known price or a reasonable default
        if manual_default_list:
            last_price = manual_default_list[-1]
        else:
            last_price = 170.0
        while len(manual_default_list) < window_size:
            manual_default_list.append(round(last_price, 2))

manual_default_text = ','.join([str(round(x, 2)) for x in manual_default_list]) if manual_default_list else ','.join(['170.00'] * window_size)

st.subheader("Manual Input (Recent Close Prices)")
manual_text = st.text_area(
    label=f"Enter recent Close prices, comma-separated (minimum {window_size} values). Auto-filled from {MANUAL_DEFAULT_START.date()}.",
    value=manual_default_text,
    height=120
)

# Predict button
if st.button("Predict — Generate & Compare"):
    # Parse manual input
    try:
        values = [float(x.strip()) for x in manual_text.split(',') if x.strip() != ""]
        if len(values) == 0:
            st.error("No numeric values found in manual input.")
            st.stop()
    except Exception as e:
        st.error(f"Couldn't parse manual input: {e}")
        st.stop()

    # Ensure at least window_size values
    if len(values) < window_size:
        # repeat sequence to reach window_size (better than random)
        repeats = (window_size // len(values)) + 1
        values = (values * repeats)[:window_size]
        st.info(f"Manual input extended to {window_size} values by repeating provided values.")

    recent_values = np.array(values[-window_size:]).astype(float)

    # 1) Predict next days from manual input
    preds = predict_next_days(model, scaler, recent_values, days_to_predict, window_size)

    # Prepare date indices
    history_dates = [manual_start + timedelta(days=i) for i in range(len(values))]
    pred_dates = [history_dates[-1] + timedelta(days=i + 1) for i in range(days_to_predict)]
    pred_x = [history_dates[-1]] + pred_dates
    pred_y = [values[-1]] + list(preds)

    # 2) Plot Manual History vs Predicted (comparison)
    fig_compare = go.Figure()

    fig_compare.add_trace(go.Scatter(
        x=history_dates,
        y=values,
        mode='lines',
        name='History (Manual Input)',
        line=dict(color='black', width=2)
    ))

    fig_compare.add_trace(go.Scatter(
        x=pred_dates,
        y=preds,
        mode='lines+markers',
        name='Predicted (from manual)',
        line=dict(color='blue', width=2)
    ))

    fig_compare.update_layout(
        title={
            'text': 'Manual History vs Predicted Future Close',
            'y': 0.95, 'x': 0.5,
            'xanchor': 'center', 'yanchor': 'top'
        },
        xaxis_title='Date',
        yaxis_title='Close Price ($)',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    st.plotly_chart(fig_compare, use_container_width=True)

    # Display predictions in a table
    preds_df = pd.DataFrame({'Predicted_Close': np.round(preds, 4)}, index=pd.to_datetime(pred_dates))
    st.subheader("Predicted Future Closes")
    st.dataframe(preds_df)

    st.markdown("---")

    # -----------------------------
    # Historical training/testing style comparison using the same model & scaler
    # -----------------------------
    st.subheader("Historical — Train/Test vs Predicted (Model evaluation)")

    hist_df = fetch_close_dataframe(HISTORICAL_TICKER, hist_start.strftime("%Y-%m-%d"), (hist_end + timedelta(days=1)).strftime("%Y-%m-%d"))
    if hist_df.empty or len(hist_df) < (window_size + 10):
        st.warning("Not enough historical data available for a meaningful train/test example. Try expanding the historical date range.")
    else:
        # Build arrays for train/test split indices (based on percentage)
        total_len = len(hist_df)
        train_size = int(total_len * (hist_train_pct / 100.0))

        # Get arrays
        close_arr = hist_df['Close'].values

        # Predict across the entire series via sliding window
        preds_all = sliding_window_predictions_on_series(model, scaler, close_arr, window_size)
        # preds_all[i] corresponds to prediction for index i+window_size in the original series

        # Build aligned true/pred arrays for the predicted portion (we will compare on the test portion)
        all_pred_index = hist_df.index[window_size:]
        all_true_values = hist_df['Close'].values[window_size:]

        # Split into train/test aligned to indices
        # train portion for plotting: use history up to train_size
        train_index = hist_df.index[:train_size]
        train_values = hist_df['Close'].values[:train_size]

        # test_index starts at train_size -> but predictions start being available at window_size, so align properly
        # We'll set test portion as indices from max(window_size, train_size) to end
        test_start_idx = max(window_size, train_size)
        test_index = hist_df.index[test_start_idx:]
        test_true = hist_df['Close'].values[test_start_idx:]
        # Prediction indices aligned: need preds_all offset by (test_start_idx - window_size)
        pred_offset = test_start_idx - window_size
        pred_for_test = preds_all[pred_offset: pred_offset + len(test_true)]

        # Compute metrics
        if show_metrics:
            mae, rmse = compute_metrics(test_true, pred_for_test)
            st.markdown(f"**Test Metrics** — MAE: {mae:.4f} — RMSE: {rmse:.4f}")

        # Build figure like your training/testing example
        fig_hist = go.Figure()

        # Training trace (black)
        fig_hist.add_trace(go.Scatter(
            x=train_index,
            y=train_values,
            mode='lines',
            name='Training',
            line=dict(color='black', width=2)
        ))

        # True test values (red) - we show from test_start_idx onward
        fig_hist.add_trace(go.Scatter(
            x=test_index,
            y=test_true,
            mode='lines',
            name='Testing (True)',
            line=dict(color='red', width=2)
        ))

        # Predicted values (blue) aligned to test_index
        fig_hist.add_trace(go.Scatter(
            x=test_index,
            y=pred_for_test,
            mode='lines',
            name='Predicted (Model)',
            line=dict(color='blue', width=2)
        ))

        fig_hist.update_layout(
            title={
                'text': 'Model Performance on Predicting AAPL Stock Closing Price (Historical)',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Date',
            yaxis_title='Close Price ($)',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        st.plotly_chart(fig_hist, use_container_width=True)

        # Optionally show a small table with first N comparisons
        compare_show_n = min(30, len(test_true))
        compare_df = pd.DataFrame({
            'Date': pd.to_datetime(test_index[:compare_show_n]),
            'True_Close': test_true[:compare_show_n],
            'Predicted_Close': np.round(pred_for_test[:compare_show_n], 4)
        }).set_index('Date')
        st.subheader(f"First {compare_show_n} test rows — True vs Predicted")
        st.dataframe(compare_df)

st.markdown("---")


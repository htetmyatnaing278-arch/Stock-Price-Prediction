# app.py
import os
import random
from datetime import datetime, timedelta, date
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
TICKER = "AAPL"
HISTORICAL_DEFAULT_START = datetime(2020, 1, 1).date()

# -----------------------------
# Helpers: load saved components
# -----------------------------
@st.cache_resource
def load_saved_components(model_path=MODEL_PATH, scaler_path=SCALER_PATH, window_path=WINDOW_PATH):
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
    """
    Return DataFrame with DatetimeIndex and 'Close' column.
    start/end: strings 'YYYY-MM-DD' or date objects
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty or 'Close' not in df.columns:
        return pd.DataFrame()
    df = df[['Close']].dropna()
    return df

@st.cache_data
def fetch_recent_closes(ticker, lookback_days_buffer, required_count):
    """
    Fetch recent historical closes using a buffer in calendar days.
    Returns the last `required_count` closes if available; otherwise returns [].
    """
    today = datetime.now().date()
    start = today - timedelta(days=lookback_days_buffer)
    df = fetch_close_dataframe(ticker, start.strftime("%Y-%m-%d"), (today + timedelta(days=1)).strftime("%Y-%m-%d"))
    if df.empty:
        return []
    closes = df['Close'].dropna().tolist()
    if len(closes) < required_count:
        return []
    return closes[-required_count:]

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
    return preds_inv

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# -----------------------------
# Streamlit UI / Layout
# -----------------------------
st.set_page_config(page_title='AAPL — LSTM Predictor (Full)', layout='wide')
st.title("AAPL Close Price — LSTM Predictor (Full)")

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
    # Mode: days or until-date or both
    st.subheader("Prediction horizon mode")
    mode_choice = st.radio("Choose horizon input mode:", ("Days (numeric)", "Until date", "Both (choose either)"))

    days_input = st.number_input("Days to predict (if using Days mode)", min_value=1, max_value=800, value=7, step=1)

    today = datetime.now().date()
    default_until = today + timedelta(days=30)
    until_date = st.date_input("Predict until date (if using Until date mode)", value=default_until, min_value=today + timedelta(days=1))

    dec31_checkbox = st.checkbox("Quick: Predict until Dec 31, 2025", value=False)
    if dec31_checkbox:
        until_date = date(2025, 12, 31)

    # Other toggles
    st.markdown("---")
    auto_fill = st.checkbox("Auto-fill manual input with latest closes (recommended)", value=True)
    show_metrics = st.checkbox("Show historical test metrics (MAE & RMSE)", value=True)

    st.markdown("---")
    st.subheader("Historical evaluation")
    hist_start = st.date_input("Historical start date", value=HISTORICAL_DEFAULT_START)
    hist_end = st.date_input("Historical end date", value=today)
    hist_train_pct = st.slider("Train split (%) for historical evaluation", min_value=50, max_value=90, value=80)

# -----------------------------
# Auto-fill manual input with last `window_size` trading-day closes (no fake defaults)
# -----------------------------
manual_default_list = []
if auto_fill:
    # use a buffer of calendar days to ensure we capture enough trading days
    lookback_buffer_days = max(60, window_size * 4)
    manual_default_list = fetch_recent_closes(TICKER, lookback_buffer_days, window_size)

manual_default_text = ""
if manual_default_list:
    manual_default_text = ",".join([str(round(x, 2)) for x in manual_default_list])

st.subheader("Manual Input (Recent Close Prices)")
st.markdown(f"Provide the most recent close prices (comma-separated). Recommended length: **{window_size}** values.")
manual_text = st.text_area(
    label="Manual prices (comma-separated):",
    value=manual_default_text,
    height=120
)

# Offer a quick 'fill from last X days' button
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Refill from latest market data"):
        lookback_buffer_days = max(120, window_size * 5)
        manual_default_list = fetch_recent_closes(TICKER, lookback_buffer_days, window_size)
        if not manual_default_list:
            st.warning("Could not fetch enough recent closes to auto-fill. You can enter values manually.")
        else:
            # replace the text area using st.experimental_rerun pattern: write to session_state and rerun
            st.session_state['_manual_default_text'] = ",".join([str(round(x, 2)) for x in manual_default_list])
            st.experimental_rerun()
with col2:
    st.write("If auto-fill fails, enter real recent closes manually (most recent last).")

# Allow session_state refill when button was used
if '_manual_default_text' in st.session_state and st.session_state['_manual_default_text']:
    manual_text = st.session_state['_manual_default_text']

# -----------------------------
# Determine final days_to_predict based on mode_choice / inputs
# -----------------------------
def compute_days_to_predict_from_inputs(mode, days_val, until_val):
    today = datetime.now().date()
    if mode == "Days (numeric)":
        return int(days_val)
    if mode == "Until date":
        delta = (until_val - today).days
        return max(1, delta)
    # Both: user can provide either — prefer until_date if it's in the future and not default
    if mode == "Both (choose either)":
        # If until_val is in the future and > days_val then prefer until_val
        delta = (until_val - today).days
        if delta >= 1:
            return max(1, delta)
        return int(days_val)
    return int(days_val)

days_to_predict = compute_days_to_predict_from_inputs(mode_choice, days_input, until_date)

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict — Generate & Compare"):
    # Parse manual input
    try:
        values = [float(x.strip()) for x in manual_text.split(',') if x.strip() != ""]
    except Exception as e:
        st.error(f"Couldn't parse manual input: {e}")
        st.stop()

    if len(values) == 0:
        st.error("Manual input is empty. Please provide the most recent close prices or enable auto-fill and refill.")
        st.stop()

    # If user provided more or less than window_size, we'll use the most recent window_size values.
    if len(values) < window_size:
        st.error(f"Manual input needs at least {window_size} values (most recent last). Provide more data or use the refill button.")
        st.stop()

    # Keep only last window_size values as model expects sequence length = window_size
    recent_values = np.array(values[-window_size:]).astype(float)

    # 1) Predict next days from manual input
    try:
        preds = predict_next_days(model, scaler, recent_values, days_to_predict, window_size)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Prepare date indices
    # Interpret manual input dates: we don't know exact trading dates the user used; assume last date is today
    last_known_date = datetime.now().date()
    # Build history_dates backwards from last_known_date
    history_dates = []
    # Place history_dates so that last element corresponds to last_known_date
    for i in range(len(values[-window_size:]) - 1, -1, -1):
        offset = len(values[-window_size:]) - 1 - i
        history_dates.append(last_known_date - timedelta(days=offset))
    # Now history_dates is ascending
    history_dates = sorted(history_dates)

    pred_dates = [history_dates[-1] + timedelta(days=i + 1) for i in range(days_to_predict)]

    # 2) Plot Manual History vs Predicted (comparison)
    fig_compare = go.Figure()

    fig_compare.add_trace(go.Scatter(
        x=history_dates,
        y=values[-window_size:],
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
            'text': f'Manual History vs Predicted Future Close — Next {days_to_predict} days',
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

    hist_df = fetch_close_dataframe(TICKER, hist_start.strftime("%Y-%m-%d"), (hist_end + timedelta(days=1)).strftime("%Y-%m-%d"))
    if hist_df.empty or len(hist_df) < (window_size + 10):
        st.warning("Not enough historical data available for a meaningful train/test example. Try expanding the historical date range.")
    else:
        total_len = len(hist_df)
        train_size = int(total_len * (hist_train_pct / 100.0))

        close_arr = hist_df['Close'].values

        # sliding window predictions across the series
        preds_all = sliding_window_predictions_on_series(model, scaler, close_arr, window_size)
        # preds_all[i] corresponds to prediction for original series index i+window_size

        # Align true/pred arrays
        all_pred_index = hist_df.index[window_size:]
        all_true_values = hist_df['Close'].values[window_size:]

        # test start index for evaluation
        test_start_idx = max(window_size, train_size)
        test_index = hist_df.index[test_start_idx:]
        test_true = hist_df['Close'].values[test_start_idx:]

        pred_offset = test_start_idx - window_size
        pred_for_test = preds_all[pred_offset: pred_offset + len(test_true)]

        if len(pred_for_test) != len(test_true):
            # alignment problem; trim to shortest
            min_len = min(len(pred_for_test), len(test_true))
            pred_for_test = pred_for_test[:min_len]
            test_true = test_true[:min_len]
            test_index = test_index[:min_len]

        if show_metrics and len(test_true) > 0:
            mae, rmse = compute_metrics(test_true, pred_for_test)
            st.markdown(f"**Test Metrics** — MAE: {mae:.4f} — RMSE: {rmse:.4f}")

        # Build training/testing style figure
        fig_hist = go.Figure()

        # Training trace (black)
        fig_hist.add_trace(go.Scatter(
            x=hist_df.index[:train_size],
            y=hist_df['Close'].values[:train_size],
            mode='lines',
            name='Training',
            line=dict(color='black', width=2)
        ))

        # True test values (red)
        fig_hist.add_trace(go.Scatter(
            x=test_index,
            y=test_true,
            mode='lines',
            name='Testing (True)',
            line=dict(color='red', width=2)
        ))

        # Predicted values (blue)
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

        # Show small comparison table
        compare_show_n = min(30, len(test_true))
        compare_df = pd.DataFrame({
            'Date': pd.to_datetime(test_index[:compare_show_n]),
            'True_Close': test_true[:compare_show_n],
            'Predicted_Close': np.round(pred_for_test[:compare_show_n], 4)
        }).set_index('Date')
        st.subheader(f"First {compare_show_n} test rows — True vs Predicted")
        st.dataframe(compare_df)

st.markdown("---")
st.write("Notes:")
st.write("- App expects your saved `lstm_aapl_model.h5`, `scaler.pkl`, and `window_size.txt` files at the top paths.")
st.write("- Manual input must include real recent closes (most recent last). Auto-fill tries to get the last `window_size` trading closes.")
st.write("- Prediction alignment assumes manual input's last item corresponds to today. If you have exact dates for historic values, replace history_dates construction accordingly.")

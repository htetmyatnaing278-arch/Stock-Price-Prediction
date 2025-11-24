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
try:
    tf.random.set_seed(random_seed)
except Exception:
    pass

# -----------------------------
# Load model, scaler, and window size
# -----------------------------
@st.cache_resource
def load_saved_components():
    model_path = 'aapl_lstm_streamlit_app/lstm_aapl_model.h5'
    scaler_path = 'aapl_lstm_streamlit_app/scaler.pkl'
    window_path = 'aapl_lstm_streamlit_app/window_size.txt'

    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    if not os.path.exists(scaler_path):
        st.error(f"Scaler file not found at: {scaler_path}")
        st.stop()
    if not os.path.exists(window_path):
        st.error(f"Window size file not found at: {window_path}")
        st.stop()

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    with open(window_path, 'r') as f:
        window_size = int(f.read().strip())

    return model, scaler, window_size

# -----------------------------
# Prediction helper function
# -----------------------------
def predict_next_days(model, scaler, recent_values, days, window_size):
    recent_values_slice = recent_values.tail(window_size)
    scaled = scaler.transform(recent_values_slice.values.reshape(-1, 1))
    scaled_list = list(scaled.flatten())

    preds_scaled = []
    for _ in range(days):
        seq = np.array(scaled_list[-window_size:]).reshape(1, window_size, 1)
        p = model.predict(seq, verbose=0)
        preds_scaled.append(p.flatten()[0])
        scaled_list.append(p.flatten()[0])

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()

# -----------------------------
# Get Latest AAPL Price (Live from Yahoo)
# -----------------------------
@st.cache_resource
def get_latest_aapl_price():
    try:
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="2d")  # fetch last 2 days just to be safe
        if hist.empty:
            raise ValueError("No data returned from Yahoo Finance")
        return float(hist['Close'].iloc[-1])
    except Exception as e:
        st.warning(f"Failed to fetch live AAPL price: {e}. Using fallback 275.0")
        return 275.0

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title='AAPL Close Price Predictor', layout='wide')
st.title('AAPL Close Price — LSTM Predictor')

model, scaler, window_size = load_saved_components()
st.success(f'Model loaded successfully — window_size = {window_size}')

st.subheader('Manual Input')

# Get latest live price
latest_price = get_latest_aapl_price()
st.info(f"Using latest AAPL close price from Yahoo Finance: **${latest_price:.2f}**")

# Generate default input history around latest price
default_values = [str(round(latest_price + random.uniform(-3, 3), 2)) for _ in range(window_size)]
manual_text = st.text_area(
    f'Enter recent Close prices (comma-separated, minimum {window_size} values):',
    value=','.join(default_values)
)

days = st.number_input('Days to predict', min_value=1, max_value=30, value=7)

if st.button('Predict'):
    try:
        values = [float(x.strip()) for x in manual_text.split(',') if x.strip()]

        if len(values) < window_size:
            st.warning(f"Input too short (only {len(values)} prices). Padding with repeats.")
            repeats = (window_size // len(values)) + 1
            values = (values * repeats)[:window_size]

        recent_values = pd.Series(values[-window_size:])
        preds = predict_next_days(model, scaler, recent_values, days, window_size)

        # Create dates for plotting
        start_date = datetime.today()
        history_dates = [start_date - timedelta(days=len(values) - i - 1) for i in range(len(values))]
        pred_dates = [history_dates[-1] + timedelta(days=i + 1) for i in range(days)]

        # Plotting
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_dates,
            y=values,
            name='Input History',
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=[history_dates[-1]] + pred_dates,
            y=[values[-1]] + list(preds),
            name=f'Predicted ({days} days)',
            mode='lines+markers',
            line=dict(color='red')
        ))

        fig.update_layout(
            title='Manual Input — Predicted Close',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        st.plotly_chart(fig, use_container_width=True)

        preds_df = pd.DataFrame({'Predicted_Close': preds}, index=pred_dates)
        st.subheader(f'Forecast for the Next {days} Days')
        st.dataframe(preds_df.style.format("{:.2f}"))

    except Exception as e:
        st.error(f'An unexpected error occurred during prediction: {e}')

st.markdown('---')

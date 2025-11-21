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
# Helper functions
# -----------------------------
def predict_next_days(model, scaler, recent_values, days, window_size):
    scaled = scaler.transform(recent_values.values.reshape(-1, 1))
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
# Get Latest AAPL Price
# -----------------------------
@st.cache_resource
def get_latest_aapl_price():
    try:
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="1d")
        price = hist['Close'].iloc[-1]
        return float(price)
    except:
        return None

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title='AAPL Close Price Predictor', layout='wide')
st.title('AAPL Close Price — LSTM Predictor')

model, scaler, window_size = load_saved_components()
st.success(f'Model loaded successfully — window_size = {window_size}')

# -----------------------------
# Manual Input with Live Price
# -----------------------------
st.subheader('Manual Input')

latest_price = get_latest_aapl_price()

if latest_price is not None:
    st.info(f"Latest AAPL closing price fetched from Yahoo Finance: **${latest_price:.2f}**")
else:
    st.warning("Could not fetch live AAPL price. Using default values 155–180.")
    latest_price = 170.0  # fallback

# Create realistic default values around the latest price
default_values = [
    str(round(latest_price + random.uniform(-3, 3), 2))
    for _ in range(window_size)
]
default_text = ','.join(default_values)

manual_text = st.text_area(
    f'Enter recent Close prices (comma-separated, minimum {window_size} values):',
    value=default_text
)

days = st.number_input('Days to predict', min_value=1, max_value=30, value=7)

if st.button('Predict'):
    try:
        values = [float(x.strip()) for x in manual_text.split(',') if x.strip()]

        if len(values) < window_size:
            repeats = (window_size // len(values)) + 1
            values = (values * repeats)[:window_size]

        recent_values = pd.Series(values)
        preds = predict_next_days(model, scaler, recent_values, days, window_size)

        pred_index = list(range(len(values), len(values) + days))
        preds_df = pd.DataFrame({'Predicted_Close': preds}, index=pred_index)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(values))), y=values,
            name='Manual history'
        ))
        fig.add_trace(go.Scatter(
            x=preds_df.index, y=preds_df['Predicted_Close'],
            name='Predicted', mode='lines+markers', line=dict(color='red'))

        fig.update_layout(title='Manual Input — Predicted Close')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(preds_df)

    except Exception as e:
        st.error(f'Input error: {e}')

st.markdown('---')

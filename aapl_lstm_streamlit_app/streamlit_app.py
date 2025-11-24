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
        return float(hist['Close'].iloc[-1])
    except:
        return None

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title='AAPL Close Price Predictor', layout='wide')
st.title('AAPL Close Price — LSTM Predictor')

model, scaler, window_size = load_saved_components()
window_size = 60  # Override for evaluation setup
st.success(f'Model loaded successfully — using window_size = {window_size}')

# -----------------------------
# Manual Input
# -----------------------------
st.subheader('Manual Input')

latest_price = get_latest_aapl_price()
if latest_price is None:
    latest_price = 170.0

# Default prices around latest price
default_values = [
    str(round(latest_price + random.uniform(-3, 3), 2))
    for _ in range(90)
]
default_text = ','.join(default_values)

manual_text = st.text_area(
    'Enter 90 recent Close prices (comma-separated):',
    value=default_text
)

if st.button('Predict'):
    try:
        values = [float(x.strip()) for x in manual_text.split(',') if x.strip()]
        if len(values) < 90:
            st.error("Please enter at least 90 values (60 for input, 30 for comparison).")
            st.stop()

        input_series = pd.Series(values[:60])
        actual_future = values[60:90]
        preds = predict_next_days(model, scaler, input_series, 30, window_size)

        # -----------------------------
        # Create date index for x-axis
        # -----------------------------
        start_date = datetime.today()
        input_dates = [start_date - timedelta(days=59 - i) for i in range(60)]
        future_dates = [input_dates[-1] + timedelta(days=i + 1) for i in range(30)]

        # -----------------------------
        # Plotting
        # -----------------------------
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=input_dates,
            y=input_series,
            name='Input History (60)',
            line=dict(color='green')
        ))

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=actual_future,
            name='Actual Future (30)',
            mode='lines+markers',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=preds,
            name='Predicted Future (30)',
            mode='lines+markers',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title='60-Day Input + 30-Day Prediction vs Actual',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display comparison DataFrame
        comparison_df = pd.DataFrame({
            'Date': future_dates,
            'Actual_Close': actual_future,
            'Predicted_Close': preds
        }).set_index('Date')
        st.dataframe(comparison_df)

    except Exception as e:
        st.error(f'Input error: {e}')

st.markdown('---')

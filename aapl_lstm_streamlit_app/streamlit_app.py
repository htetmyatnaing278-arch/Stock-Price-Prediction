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

@st.cache_data
def fetch_aapl_close_prices(start_date, num_days):
    try:
        end_date = start_date + timedelta(days=num_days * 2)  # buffer for weekends/holidays
        df = yf.download("AAPL", start=start_date, end=end_date)
        closes = df['Close'].dropna().tolist()
        return closes[:num_days]
    except Exception:
        return []

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title='AAPL Close Price Predictor', layout='wide')
st.title('AAPL Close Price — LSTM Predictor')

model, scaler, window_size = load_saved_components()
st.success(f'Model loaded successfully — window_size = {window_size}')

# -----------------------------
# Manual Input
# -----------------------------
st.subheader('Manual Input')

start_date = st.date_input("Start date of the first Close price", datetime(2025, 1, 1))
days = st.number_input('Days to predict', min_value=1, max_value=30, value=7)

# Fetch real close prices from start_date
real_prices = fetch_aapl_close_prices(start_date, window_size)
if len(real_prices) < window_size:
    st.warning(f"Only {len(real_prices)} trading days found from {start_date}. Filling remaining with random values.")
    while len(real_prices) < window_size:
        real_prices.append(round(random.uniform(160, 180), 2))

default_text = ','.join([str(round(p, 2)) for p in real_prices])

manual_text = st.text_area(
    f'Enter recent Close prices (comma-separated, minimum {window_size} values):',
    value=default_text
)

if st.button('Predict'):
    try:
        values = [float(x.strip()) for x in manual_text.split(',') if x.strip()]
        if len(values) < window_size:
            repeats = (window_size // len(values)) + 1
            values = (values * repeats)[:window_size]

        recent_values = pd.Series(values)
        preds = predict_next_days(model, scaler, recent_values, days, window_size)

        # -----------------------------
        # Create date index for x-axis
        # -----------------------------
        history_dates = [start_date + timedelta(days=i) for i in range(len(values))]
        pred_dates = [history_dates[-1] + timedelta(days=i + 1) for i in range(days)]

        # Combine dates for plotting connection
        pred_x = [history_dates[-1]] + pred_dates
        pred_y = [values[-1]] + list(preds)

        # -----------------------------
        # Plotting
        # -----------------------------
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=history_dates,
            y=values,
            name='Manual history',
            line=dict(color='green')
        ))

        fig.add_trace(go.Scatter(
            x=pred_x,
            y=pred_y,
            name='Predicted',
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

        # Display predicted values in a DataFrame
        preds_df = pd.DataFrame({'Predicted_Close': preds}, index=pred_dates)
        st.dataframe(preds_df)

    except Exception as e:
        st.error(f'Input error: {e}')

st.markdown('---')

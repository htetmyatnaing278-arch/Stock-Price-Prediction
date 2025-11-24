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
def predict_red_line(model, scaler, green_values, n_red):
    """
    Predict next `n_red` days using only green_values as history.
    This aligns predictions with the red line (actual last 30 days).
    """
    # Ensure green_values >= window_size
    if len(green_values) < window_size:
        pad_len = window_size - len(green_values)
        green_values = [green_values[0]] * pad_len + green_values

    scaled = scaler.transform(np.array(green_values).reshape(-1, 1))
    scaled_list = list(scaled.flatten())

    preds_scaled = []
    for _ in range(n_red):
        seq = np.array(scaled_list[-window_size:]).reshape(1, window_size, 1)
        p = model.predict(seq, verbose=0)
        preds_scaled.append(p.flatten()[0])
        scaled_list.append(p.flatten()[0])

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()

# -----------------------------
# Get latest AAPL price
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
st.success(f'Model loaded successfully — window_size = {window_size}')

st.subheader('Manual Input')

# Get latest price for default
latest_price = get_latest_aapl_price()
if latest_price is None:
    latest_price = 170.0

# Default values for manual input
default_values = [str(round(latest_price + random.uniform(-3, 3), 2)) for _ in range(window_size)]
manual_text = st.text_area(
    f'Enter {window_size} recent Close prices (comma-separated):',
    value=','.join(default_values)
)

# Days to predict
days = st.number_input('Days to predict', min_value=1, max_value=30, value=7)

# -----------------------------
# Prediction and Plot
# -----------------------------
if st.button('Predict'):
    try:
        # Parse manual input
        values = [float(x.strip()) for x in manual_text.split(',') if x.strip()]
        if len(values) < window_size:
            repeats = (window_size // len(values)) + 1
            values = (values * repeats)[:window_size]

        recent_values = pd.Series(values)

        # Split: green = first 60, red = last 30
        green_values = recent_values[:60].tolist()
        red_values = recent_values[60:90].tolist()  # actual last 30

        # Predict using only green_values for 30 days to match red
        predicted_red = predict_red_line(model, scaler, green_values, n_red=30)

        # -----------------------------
        # Create dates
        # -----------------------------
        today = datetime.today()
        green_dates = [today - timedelta(days=90 - i) for i in range(60)]
        red_dates = [today - timedelta(days=30 - i) for i in range(30)]

        # -----------------------------
        # Plotting
        # -----------------------------
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=green_dates,
            y=green_values,
            name='History (60 days)',
            line=dict(color='green'),
            mode='lines+markers'
        ))
        fig.add_trace(go.Scatter(
            x=red_dates,
            y=red_values,
            name='Actual last 30 days',
            line=dict(color='red'),
            mode='lines+markers'
        ))
        fig.add_trace(go.Scatter(
            x=red_dates,
            y=predicted_red,
            name='Predicted (from 60 days)',
            line=dict(color='skyblue'),
            mode='lines+markers'
        ))

        fig.update_layout(
            title='AAPL Close Price Prediction (Comparison)',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            yaxis=dict(tickformat="$,.2f"),
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display predicted future prices
        preds_df = pd.DataFrame({'Predicted_Close ($)': predicted_red}, index=red_dates)
        st.subheader('Predicted Prices vs Actual Last 30 Days')
        st.dataframe(preds_df.style.format("${:.2f}"))

    except Exception as e:
        st.error(f'An unexpected error occurred: {e}')

st.markdown('---')

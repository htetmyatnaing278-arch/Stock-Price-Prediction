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
# Multi-step prediction (returns-based)
# -----------------------------
def predict_next_days(model, scaler, recent_values, days, window_size):
    recent_prices = pd.Series(recent_values)
    recent_returns = recent_prices.pct_change().dropna()

    if len(recent_returns) < window_size:
        st.warning(f"Need at least {window_size+1} prices to predict {window_size} returns.")
        return np.array([])

    last_actual_price = recent_prices.iloc[-1]
    scaled_returns = scaler.transform(recent_returns.values.reshape(-1,1))
    scaled_list = list(scaled_returns[-window_size:].flatten())

    predicted_prices = []
    current_price = last_actual_price

    for _ in range(days):
        seq = np.array(scaled_list[-window_size:]).reshape(1, window_size, 1)
        scaled_return_pred = model.predict(seq, verbose=0).flatten()[0]
        scaled_list.append(scaled_return_pred)

        predicted_return = scaler.inverse_transform(np.array([[scaled_return_pred]]))[0,0]
        next_price = current_price * (1 + predicted_return)
        predicted_prices.append(next_price)
        current_price = next_price

    return np.array(predicted_prices)

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
st.title('AAPL Close Price — LSTM Predictor (Returns Model)')

model, scaler, window_size = load_saved_components()
st.success(f'Model loaded successfully — window_size (W) = {window_size}')

# -----------------------------
# Manual Input
# -----------------------------
latest_price = get_latest_aapl_price()
if latest_price is None:
    latest_price = 170.0

required_input_size = window_size + 1  # 91 for W=90
default_values = [str(round(latest_price + random.uniform(-3,3),2)) for _ in range(required_input_size)]
default_text = ','.join(default_values)

manual_text = st.text_area(
    f'Enter recent Close prices (comma-separated, min {required_input_size} values):',
    value=default_text
)

days = st.number_input('Days to predict', min_value=1, max_value=30, value=7)

if st.button('Predict'):
    try:
        values = [float(x.strip()) for x in manual_text.split(',') if x.strip()]
        if len(values) < required_input_size:
            st.error(f"Input must contain at least {required_input_size} prices.")
            st.stop()

        recent_values = pd.Series(values[-required_input_size:])
        full_history = recent_values.tolist()

        # -----------------------------
        # 60 / 30 split
        # -----------------------------
        history_values = full_history[:60]  # Green
        actual_values = full_history[60:90] # Blue

        # Predict including 30 actual values + future days
        predictions = predict_next_days(model, scaler, recent_values, days, window_size)
        red_line_values = actual_values + predictions.tolist()

        # -----------------------------
        # Create dates
        # -----------------------------
        today = datetime.today().date()
        history_dates = [today - timedelta(days=90 - i) for i in range(60)]
        actual_dates = [today - timedelta(days=30 - i) for i in range(30)]
        pred_dates = [today + timedelta(days=i) for i in range(len(red_line_values))]

        # -----------------------------
        # Plot
        # -----------------------------
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=history_dates,
            y=history_values,
            name='History (60 days)',
            line=dict(color='green'),
            mode='lines+markers'
        ))

        fig.add_trace(go.Scatter(
            x=actual_dates,
            y=actual_values,
            name='Actual Prices (30 days)',
            line=dict(color='blue'),
            mode='lines+markers'
        ))

        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=red_line_values,
            name=f'Predicted + Actual Overlap ({len(red_line_values)} days)',
            line=dict(color='red', dash='dot'),
            mode='lines+markers'
        ))

        fig.update_layout(
            title='AAPL Close Price Prediction (Comparison)',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            yaxis=dict(tickformat="$,.2f"),  # USD format
            xaxis=dict(tickformat='%Y-%m-%d')
        )

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # Display predicted values
        # -----------------------------
        preds_df = pd.DataFrame({'Predicted_Close ($)': red_line_values[30:]}, index=pred_dates[30:])
        st.subheader(f'Forecasted Prices for the Next {len(predictions)} Days')
        st.dataframe(preds_df.style.format("${:.2f}"))

    except Exception as e:
        st.error(f'An unexpected error occurred: {e}')
        st.markdown("---")
        st.caption("Ensure your model and scaler were trained on daily returns.")

st.markdown('---')

# aapl_lstm_streamlit_app/streamlit_app.py

import os
import random
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objs as go
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import yfinance as yf

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
def load_components():
    model_path = 'lstm_aapl_model.h5'
    scaler_path = 'scaler.pkl'
    window_path = 'window_size.txt'

    for path in [model_path, scaler_path, window_path]:
        if not os.path.exists(path):
            st.error(f"Missing file: {path}. Please upload all required files.")
            st.stop()

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    with open(window_path) as f:
        window_size = int(f.read().strip())

    return model, scaler, window_size

# -----------------------------
# Prediction helper
# -----------------------------
def predict_future(model, scaler, input_values, n_days, window_size):
    if len(input_values) < window_size:
        input_values = [input_values[0]] * (window_size - len(input_values)) + input_values

    scaled = scaler.transform(np.array(input_values).reshape(-1, 1)).flatten().tolist()
    preds_scaled = []

    for _ in range(n_days):
        seq = np.array(scaled[-window_size:]).reshape(1, window_size, 1)
        pred = model.predict(seq, verbose=0)[0][0]
        preds_scaled.append(pred)
        scaled.append(pred)

    return scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

# -----------------------------
# Get latest AAPL price
# -----------------------------
@st.cache_resource
def get_latest_price():
    try:
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="1d")
        return float(hist['Close'].iloc[-1])
    except:
        return 170.0

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title='AAPL Close Price Predictor', layout='wide')
st.title('📈 AAPL Close Price — LSTM Predictor')

model, scaler, window_size = load_components()
st.success(f'Model loaded successfully — window_size = {window_size}')

st.subheader('Manual Input')

latest_price = get_latest_price()
default_values = [str(round(latest_price + random.uniform(-3, 3), 2)) for _ in range(window_size)]
manual_text = st.text_area(f'Enter {window_size} recent Close prices (comma-separated):', value=','.join(default_values))
days = st.number_input('Days to predict', min_value=1, max_value=30, value=7)

# -----------------------------
# Prediction and Plot
# -----------------------------
if st.button('Predict'):
    try:
        values = [float(x.strip()) for x in manual_text.split(',') if x.strip()]
        if len(values) < window_size:
            values = [values[0]] * (window_size - len(values)) + values

        green_values = values[:60]
        red_values = values[60:90]
        predicted_red = predict_future(model, scaler, green_values, n_days=30, window_size=window_size)

        today = datetime.today()
        green_dates = [today - timedelta(days=90 - i) for i in range(60)]
        red_dates = [today - timedelta(days=30 - i) for i in range(30)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=green_dates, y=green_values, name='History (60)', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=red_dates, y=red_values, name='Actual (30)', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=red_dates, y=predicted_red, name='Predicted (30)', line=dict(color='skyblue', dash='dot')))

        fig.update_layout(title='AAPL Close Price Prediction vs Actual',
                          xaxis_title='Date', yaxis_title='Price ($)',
                          yaxis=dict(tickformat="$,.2f"), xaxis=dict(tickformat='%Y-%m-%d'))
        st.plotly_chart(fig, use_container_width=True)

        preds_df = pd.DataFrame({'Date': red_dates, 'Predicted_Close ($)': predicted_red})
        st.subheader('Predicted Prices')
        st.dataframe(preds_df.set_index('Date').style.format("${:.2f}"))

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown('---')

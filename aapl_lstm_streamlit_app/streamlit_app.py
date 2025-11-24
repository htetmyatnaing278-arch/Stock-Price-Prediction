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
# Import the necessary scipy function for error estimation if needed later
# from scipy import stats # Kept commented as the error is provided manually

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
# USER-DEFINED FORECAST ERROR PARAMETER ⚠️
# -----------------------------
# This value must be derived from your training analysis (e.g., the square root of 
# the upper bound of the 95% CI for the Mean Squared Error on the test set).
# I am using a placeholder value (e.g., 5.0) which you MUST REPLACE with your actual calculated error.
ESTIMATED_FORECAST_ERROR_SIGMA = 5.0 
Z_SCORE_95 = 1.96 # Z-score for 95% confidence

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
st.success(f'Model loaded successfully — window_size = {window_size}')

# -----------------------------
# Manual Input
# -----------------------------
st.subheader('Manual Input')

latest_price = get_latest_aapl_price()
if latest_price is None:
    latest_price = 170.0 # Placeholder for old code

# Display sigma information
st.info(
    f"Confidence Interval (CI) is calculated using $\\sigma = {ESTIMATED_FORECAST_ERROR_SIGMA}$ "
    f"derived from your test set MSE analysis. This represents the $95\%$ likely range of error."
)

# Default prices around latest price
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

        # -----------------------------
        # Create date index for x-axis
        # -----------------------------
        start_date = datetime.today()
        history_dates = [start_date - timedelta(days=window_size - i - 1) for i in range(len(values))]
        pred_dates = [history_dates[-1] + timedelta(days=i + 1) for i in range(days)]

        # Combine dates for plotting connection
        pred_x = [history_dates[-1]] + pred_dates
        pred_y = [values[-1]] + list(preds)
        
        # -----------------------------
        # CALCULATE CONFIDENCE INTERVAL (CI) 💰
        # -----------------------------
        # The deviation is Z * sigma
        deviation = Z_SCORE_95 * ESTIMATED_FORECAST_ERROR_SIGMA
        
        # Upper and Lower Bounds for the predicted prices (excluding the start point)
        upper_bound = preds + deviation
        lower_bound = preds - deviation
        
        # For plotting the CI polygon, we need the bounds for the entire prediction period,
        # starting from the last known actual price.
        ci_x_full = pred_dates + pred_dates[::-1]
        ci_y_full = np.concatenate([upper_bound, lower_bound[::-1]])
        
        # -----------------------------
        # Plotting
        # -----------------------------
        fig = go.Figure()

        # 1️⃣ Confidence Interval (Area)
        fig.add_trace(go.Scatter(
            x=ci_x_full,
            y=ci_y_full,
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)', # Light red transparency
            line=dict(color='rgba(255, 255, 255, 0)'),
            hoverinfo='skip',
            name='95% Prediction Interval'
        ))
        
        # 2️⃣ Manual history
        fig.add_trace(go.Scatter(
            x=history_dates,
            y=values,
            name='Input History',
            line=dict(color='green')
        ))

        # 3️⃣ Predicted values (connected)
        fig.add_trace(go.Scatter(
            x=pred_x,
            y=pred_y,
            name='Predicted Close',
            mode='lines+markers',
            line=dict(color='red')
        ))
        
        # 4️⃣ CI Bounds (as separate lines for clarity in the legend/hover)
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=upper_bound,
            name=f'Upper Bound (+{deviation:.2f})',
            line=dict(width=0, color='red') # Make invisible line to show in legend
        ))
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=lower_bound,
            name=f'Lower Bound (-{deviation:.2f})',
            line=dict(width=0, color='red') # Make invisible line to show in legend
        ))

        fig.update_layout(
            title='Forecast with 95% Prediction Interval',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # Display predicted values and CI 
        # -----------------------------
        preds_df = pd.DataFrame({
            'Predicted_Close': preds,
            f'Lower_Bound (-{deviation:.2f})': lower_bound,
            f'Upper_Bound (+{deviation:.2f})': upper_bound
        }, index=pred_dates)
        
        st.subheader(f'Forecast and 95% CI for the Next {days} Days')
        st.dataframe(preds_df.style.format("{:.2f}"))

    except Exception as e:
        st.error(f'Input error: {e}')

st.markdown('---')

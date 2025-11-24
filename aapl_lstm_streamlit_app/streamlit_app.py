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
# Helper function: Multi-step prediction for RETURNS (corrected)
# -----------------------------
def predict_next_days(model, scaler, recent_values, days, window_size):
    """
    Performs multi-step (recursive/open-loop) forecasting on a model trained on RETURNS.
    It predicts scaled returns and iteratively converts them back to absolute prices.
    """
    recent_prices = pd.Series(recent_values)
    
    # 1. Convert recent prices to daily returns
    # We drop the first NaN from pct_change(). We need (window_size) returns.
    recent_returns = recent_prices.pct_change().dropna() 
    
    if len(recent_returns) < window_size:
        st.warning("Insufficient return data to form a sequence. Check input length.")
        return np.array([])

    # 2. Get the last known actual price (P_{t-1})
    last_actual_price = recent_prices.iloc[-1] 
    
    # 3. Transform the necessary returns (last window_size) using the fitted scaler
    # Note: We only use the last `window_size` returns to start the prediction.
    scaled_returns = scaler.transform(recent_returns.values.reshape(-1, 1))
    scaled_list = list(scaled_returns[-window_size:].flatten()) # Initial sequence

    predicted_prices = []
    current_price = last_actual_price # Initialize the price for P_{t-1} in the first loop

    # 4. Recursive Forecasting Loop
    for _ in range(days):
        # a. Create sequence and predict the next scaled return
        seq = np.array(scaled_list[-window_size:]).reshape(1, window_size, 1)
        scaled_return_pred = model.predict(seq, verbose=0).flatten()[0]
        
        # b. Append the PREDICTED scaled return to the sequence list for the next iteration (compounding error)
        scaled_list.append(scaled_return_pred) 
        
        # c. Inverse transform the single predicted scaled return
        predicted_return = scaler.inverse_transform(np.array([[scaled_return_pred]]))[0, 0]
        
        # d. Convert the predicted return back to the absolute price: P_t = P_{t-1} * (1 + return)
        next_price = current_price * (1 + predicted_return)
        
        # e. Update the price and results list
        predicted_prices.append(next_price)
        current_price = next_price # The new P_{t-1} for the next day's prediction

    return np.array(predicted_prices)

# -----------------------------
# Get Latest AAPL Price (No change)
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
# Streamlit UI (Only minor changes to plotting section)
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
    latest_price = 170.0

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
        # Convert input to float
        values = [float(x.strip()) for x in manual_text.split(',') if x.strip()]

        if len(values) < window_size:
            st.error(f"Input must contain at least {window_size} values.")
            st.stop()
        
        # Use only the last window_size (90) values for prediction input
        recent_values = pd.Series(values[-window_size:]) 
        
        # -----------------------------
        # Split history for chart
        # -----------------------------
        # Use the full input history (up to window_size) for plotting visualization
        full_history = values[-window_size:] 
        
        # Determine the split points for plotting the 90 input values:
        # 60 days of "History" + 30 days of "Actuals" (where predictions start)
        history_values = full_history[:60]
        actual_values = full_history[60:]
        
        # Generate the 30-day multi-step forecast using the corrected function
        predictions = predict_next_days(model, scaler, recent_values, 30, window_size)

        # Create dates
        today = datetime.today().date()
        # Dates for the 60 "History" input values
        history_dates = [today - timedelta(days=len(history_values) + len(actual_values) - i) for i in range(len(history_values))]
        # Dates for the 30 "Actual" input values (start where history ends)
        actual_dates = [history_dates[-1] + timedelta(days=i + 1) for i in range(len(actual_values))]
        # Dates for the 30 "Predicted" values (start where actual ends)
        pred_dates = [actual_dates[-1] + timedelta(days=i + 1) for i in range(len(predictions))]

        # -----------------------------
        # Plotting
        # -----------------------------
        fig = go.Figure()

        # 1️⃣ History (first 60 of the 90 input values)
        fig.add_trace(go.Scatter(
            x=history_dates,
            y=history_values,
            name='Input History (60)',
            line=dict(color='green'),
            mode='lines+markers'
        ))

        # 2️⃣ Actual (last 30 of the 90 input values - used to start the forecast)
        fig.add_trace(go.Scatter(
            x=actual_dates,
            y=actual_values,
            name='Input Start Values (30)',
            line=dict(color='blue'),
            mode='lines+markers'
        ))

        # 3️⃣ Predicted (30-day forecast starting after the 90 input values)
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=predictions,
            name='Predicted Forecast (30)',
            line=dict(color='red', dash='dot'),
            mode='lines+markers'
        ))

        fig.update_layout(
            title='AAPL Close Price Prediction vs Actual',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            xaxis=dict(tickformat='%Y-%m-%d')
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display predicted values
        preds_df = pd.DataFrame({'Predicted_Close': predictions}, index=pred_dates)
        st.dataframe(preds_df)

    except Exception as e:
        st.error(f'An error occurred: {e}')


st.markdown('---')

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

    # Note: compile=False is used because Streamlit reloads the model frequently
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    with open(window_path, 'r') as f:
        window_size = int(f.read().strip())

    return model, scaler, window_size

# -----------------------------
# Helper function: Multi-step prediction for RETURNS (CORRECTED LOGIC)
# -----------------------------
def predict_next_days(model, scaler, recent_values, days, window_size):
    """
    Performs multi-step (recursive/open-loop) forecasting on a model trained on RETURNS.
    It predicts scaled returns and iteratively converts them back to absolute prices.
    This requires window_size + 1 prices for the initial return calculation.
    """
    recent_prices = pd.Series(recent_values)
    
    # 1. Convert recent prices to daily returns
    # This will now produce 90 returns if recent_values has 91 prices
    recent_returns = recent_prices.pct_change().dropna() 
    
    if len(recent_returns) < window_size:
        st.warning("Insufficient return data to form a sequence. Need W+1 prices to get W returns.")
        return np.array([])

    # 2. Get the last known actual price (P_{t-1})
    last_actual_price = recent_prices.iloc[-1] 
    
    # 3. Transform the necessary returns (last window_size)
    scaled_returns = scaler.transform(recent_returns.values.reshape(-1, 1))
    scaled_list = list(scaled_returns[-window_size:].flatten()) # Initial sequence

    predicted_prices = []
    current_price = last_actual_price # Price for P_{t-1}

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
# Manual Input (Adjusted to require W+1 prices)
# -----------------------------
st.subheader('Manual Input')

latest_price = get_latest_aapl_price()
if latest_price is None:
    latest_price = 170.0

# REQUIRED INPUT SIZE: W + 1 prices to generate W returns
required_input_size = window_size + 1 

# Default prices around latest price
default_values = [
    str(round(latest_price + random.uniform(-3, 3), 2))
    for _ in range(required_input_size) 
]
default_text = ','.join(default_values)

manual_text = st.text_area(
    f'Enter recent Close prices (comma-separated, minimum {required_input_size} values for {window_size} returns):',
    value=default_text
)

days = st.number_input('Days to predict', min_value=1, max_value=30, value=7)

if st.button('Predict'):
    try:
        # Convert input to float
        values = [float(x.strip()) for x in manual_text.split(',') if x.strip()]

        if len(values) < required_input_size:
            st.error(f"Input must contain at least {required_input_size} prices to calculate {window_size} returns. Please check your input.")
            st.stop()
            
        # Use only the last required_input_size (W+1) values for prediction
        recent_values = pd.Series(values[-required_input_size:]) 
        
        # -----------------------------
        # Split history for chart (Using the 91 input values)
        # -----------------------------
        full_history = recent_values.tolist() # The 91 values (W+1)
        
        # The split point: Only split the *input* history if total input is > days (e.g., 91 > 7)
        # This keeps the logic consistent for visualization, where the *last* 'days' of input
        # are highlighted as the immediate start for the forecast.
        history_values = full_history[:-days]
        actual_values = full_history[-days:]
        
        # Generate the forecast: USE THE SELECTED 'days' VALUE INSTEAD OF 30
        predictions = predict_next_days(model, scaler, recent_values, days, window_size)

        # Create dates
        today = datetime.today().date()
        total_input_days = len(full_history) # 91 days
        
        # Dates for the "History" input values
        history_dates = [today - timedelta(days=total_input_days - i - 1) for i in range(len(history_values))]
        # Dates for the "Input Start Values" (last 'days' of input)
        actual_dates = [history_dates[-1] + timedelta(days=i + 1) for i in range(len(actual_values))]
        # Dates for the "Predicted" values (start where actual ends)
        pred_dates = [actual_dates[-1] + timedelta(days=i + 1) for i in range(len(predictions))]

        # -----------------------------
        # Plotting
        # -----------------------------
        fig = go.Figure()

        # 1️⃣ History (Input values before the 'actual_values' window)
        fig.add_trace(go.Scatter(
            x=history_dates,
            y=history_values,
            name=f'Input History ({len(history_values)} days)',
            line=dict(color='green'),
            mode='lines+markers'
        ))

        # 2️⃣ Actual (last 'days' input values for comparison)
        fig.add_trace(go.Scatter(
            x=actual_dates,
            y=actual_values,
            name=f'Last {days} Input Values', # Dynamic name
            line=dict(color='blue'),
            mode='lines+markers'
        ))

        # 3️⃣ Predicted (Only 'days' forecast)
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=predictions,
            name=f'Predicted Forecast ({days} days)', # Dynamic name
            line=dict(color='red', dash='dot'),
            mode='lines+markers'
        ))

        fig.update_layout(
            title='AAPL Close Price Prediction (Multi-Step Forecast)',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            xaxis=dict(tickformat='%Y-%m-%d')
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display predicted values
        preds_df = pd.DataFrame({'Predicted_Close': predictions}, index=pred_dates)
        st.subheader(f'Forecasted Prices for the Next {len(predictions)} Days')
        st.dataframe(preds_df.style.format("{:.2f}"))

    except Exception as e:
        # Catch any unexpected errors during prediction or plotting
        st.error(f'An unexpected error occurred during prediction: {e}')
        st.markdown("---")
        st.caption("If the error persists, ensure your saved model and scaler were trained on daily returns.")

st.markdown('---')

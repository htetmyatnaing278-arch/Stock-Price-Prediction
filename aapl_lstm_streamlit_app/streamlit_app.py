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
tf.random.set_seed(random_seed)

# -----------------------------
# Load model, scaler, and window size
# -----------------------------
@st.cache_resource
def load_components():
    model_path = 'aapl_lstm_streamlit_app/lstm_aapl_model.h5'
    scaler_path = 'aapl_lstm_streamlit_app/scaler.pkl'
    window_path = 'aapl_lstm_streamlit_app/window_size.txt'

    if not all(map(os.path.exists, [model_path, scaler_path, window_path])):
        st.error("Missing model, scaler, or window size file.")
        st.stop()

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    with open(window_path, 'r') as f:
        window_size = int(f.read().strip())

    return model, scaler, window_size

# -----------------------------
# Prediction helper
# -----------------------------
def predict_sequence(model, scaler, input_values, n_days, window_size):
    if len(input_values) < window_size:
        input_values = [input_values[0]] * (window_size - len(input_values)) + input_values

    scaled = scaler.transform(np.array(input_values).reshape(-1, 1)).flatten()
    preds_scaled = []

    for _ in range(n_days):
        seq = np.array(scaled[-window_size:]).reshape(1, window_size, 1)
        pred = model.predict(seq, verbose=0).flatten()[0]
        preds_scaled.append(pred)
        scaled = np.append(scaled, pred)

    return scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

# -----------------------------
# Get latest AAPL price
# -----------------------------
@st.cache_resource
def get_latest_price():
    try:
        return float(yf.Ticker("AAPL").history(period="1d")['Close'].iloc[-1])
    except:
        return 170.0

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title='AAPL Close Price Predictor', layout='wide')
st.title('📈 AAPL Close Price — LSTM Predictor')

model, scaler, window_size = load_components()
st.success(f'Model loaded successfully — window_size = {window_size}')

# -----------------------------
# Manual Input
# -----------------------------
st.subheader('Manual Input')
latest_price = get_latest_price()
default_values = [str(round(latest_price + random.uniform(-3, 3), 2)) for _ in range(window_size)]
manual_text = st.text_area(f'Enter {window_size} recent Close prices (comma-separated):', value=','.join(default_values))
days = st.number_input('Days to predict', min_value=1, max_value=30, value=7)

# -----------------------------
# Prediction and Plot
# -----------------------------
# ... (Previous code remains the same up to the 'Prediction and Plot' section)

# -----------------------------
# Prediction and Plot
# -----------------------------
if st.button('Predict'):
    try:
        values = [float(x.strip()) for x in manual_text.split(',') if x.strip()]
        if len(values) < window_size:
            values = (values * ((window_size // len(values)) + 1))[:window_size]

        recent_series = pd.Series(values)
        green_values = recent_series[:60].tolist()
        red_values = recent_series[60:90].tolist()

        predicted_red = predict_sequence(model, scaler, green_values, n_days=30, window_size=window_size)

        # -----------------------------
        # Dates
        # -----------------------------
        today = datetime.today()
        green_dates = [today - timedelta(days=90 - i) for i in range(60)]
        red_dates = [today - timedelta(days=30 - i) for i in range(30)]

        # -----------------------------
        # CI Calculation (Required for the table)
        # -----------------------------
        if len(red_values) == len(predicted_red):
            from scipy import stats
            
            # 1. Calculate Residuals and Squared Error
            residual = np.array(red_values) - np.array(predicted_red)
            squared_error = residual ** 2
            mean_squared_error = squared_error.mean()
            
            # 2. Calculate the 95% CI for the Mean Squared Error (MSE)
            ci_level = 0.95
            mse_ci_low, mse_ci_high = stats.t.interval(
                ci_level, 
                len(squared_error) - 1,
                loc=mean_squared_error,
                scale=stats.sem(squared_error)
            )
            
            # 3. Convert CI for MSE back to CI for RMSE
            rmse_ci_low = np.sqrt(max(0, mse_ci_low)) 
            rmse_ci_high = np.sqrt(mse_ci_high)
            
            rmse_ci_95 = (rmse_ci_low, rmse_ci_high)
            
            st.subheader('Prediction Error Analysis')
            st.info(
                f'The **Root Mean Squared Error (RMSE)** for the last 30 days of actual vs. predicted prices is in the **95% Confidence Interval** of **${rmse_ci_95[0]:.2f}** to **${rmse_ci_95[1]:.2f}**.'
            )
        else:
            # Fallback if prediction length is mismatched (shouldn't happen here, but good practice)
            rmse_ci_low = 0.0
            rmse_ci_high = 0.0
        
        # -----------------------------
        # Plot (Unchanged)
        # -----------------------------
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=green_dates, y=green_values, name='History (60 days)', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=red_dates, y=red_values, name='Actual last 30 days', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=red_dates, y=predicted_red, name='Predicted (from 60 days)', line=dict(color='skyblue')))

        fig.update_layout(
            title='AAPL Close Price Prediction vs Actual',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            yaxis=dict(tickformat="$,.2f"),
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # Table (NEW/MODIFIED)
        # -----------------------------
        st.subheader('Predicted Prices with 95% Confidence Interval')
        
        # Apply the RMSE CI bounds to the predicted values to create the range
        predicted_low = predicted_red - rmse_ci_high 
        predicted_high = predicted_red + rmse_ci_high
        
        # Create the DataFrame
        preds_df = pd.DataFrame({
            'Actual Price ($)': red_values,
            'Predicted Price ($)': predicted_red,
            'CI 95% Lower Bound ($)': predicted_low,
            'CI 95% Upper Bound ($)': predicted_high
        }, index=red_dates)
        
        # Display the DataFrame with formatting
        st.dataframe(preds_df.style.format("${:.2f}"))

    except Exception as e:
        st.error(f'Error: {e}')

st.markdown('---')

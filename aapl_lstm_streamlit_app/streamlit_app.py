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
    """
    Performs multi-step (recursive/open-loop) forecasting on a model trained on prices.
    """
    # Use only the last `window_size` values needed for the first prediction
    recent_values_slice = recent_values.tail(window_size)
    
    scaled = scaler.transform(recent_values_slice.values.reshape(-1, 1))
    scaled_list = list(scaled.flatten())

    preds_scaled = []
    for _ in range(days):
        # The sequence for prediction is always the last 'window_size' values in the scaled_list
        seq = np.array(scaled_list[-window_size:]).reshape(1, window_size, 1)
        p = model.predict(seq, verbose=0)
        
        # Append the PREDICTED scaled value to the sequence list for the next iteration
        preds_scaled.append(p.flatten()[0])
        scaled_list.append(p.flatten()[0])

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()

@st.cache_data(ttl=600) # Cache for 10 minutes
def get_aapl_historical_data(window_size):
    """
    Fetches the necessary historical close price data for prediction.
    We fetch 1.5 times the window_size to ensure we get enough days,
    accounting for non-trading days (weekends/holidays).
    """
    ticker_symbol = "AAPL"
    # Fetch enough data to cover the window size, plus some buffer
    required_history_days = int(window_size * 1.5)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=required_history_days)
    
    try:
        # Download data
        data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            st.warning("Yahoo Finance returned empty data for the specified period.")
            return pd.Series(), "Data Unavailable"

        # Extract the 'Close' prices and ensure we have at least 'window_size' data points
        close_prices = data['Close'].tail(window_size)

        if len(close_prices) < window_size:
            st.warning(f"Only found {len(close_prices)} trading days in the history. Need {window_size} to predict.")
            # Pad with the most recent price if necessary to meet the minimum length
            if not close_prices.empty:
                 last_price = close_prices.iloc[-1]
                 padding = [last_price] * (window_size - len(close_prices))
                 padded_prices = pd.concat([pd.Series(padding), close_prices], ignore_index=True)
                 return padded_prices.iloc[-window_size:], "Padded Data"
            else:
                 return pd.Series(), "Data Unavailable"
        
        return close_prices, "Live Data"

    except Exception as e:
        st.error(f"Error fetching data from Yahoo Finance: {e}")
        return pd.Series(), "API Error"


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title='AAPL Close Price Predictor', layout='wide')
st.title('AAPL Close Price — LSTM Predictor 📈')

model, scaler, window_size = load_saved_components()
st.success(f'Model loaded successfully — window_size (W) = {window_size}')

# -----------------------------
# Data Source Selection
# -----------------------------
st.subheader('Data Input Source')

# Fetch live historical data
live_data_series, fetch_status = get_aapl_historical_data(window_size)
live_data_values = live_data_series.tolist()
live_data_text = ','.join([f"{x:.2f}" for x in live_data_values])

data_source = st.radio(
    "Choose Input Method:",
    ('Use Live AAPL Data', 'Enter Custom Prices Manually'),
    index=0 if fetch_status == "Live Data" else 1 # Default to live if successful
)

# -----------------------------
# Manual Input or Live Data Display
# -----------------------------
if data_source == 'Enter Custom Prices Manually':
    st.markdown("---")
    latest_price = live_data_series.iloc[-1] if not live_data_series.empty else 275.0
    st.info(f"The last price will be used as the starting point. Current market price estimate: **${latest_price:.2f}**")

    # Generate default values centered around the latest known price
    default_values = [
        str(round(latest_price + random.uniform(-3, 3), 2))
        for _ in range(window_size)
    ]
    default_text = ','.join(default_values)
    
    input_text = st.text_area(
        f'Enter recent Close prices (comma-separated, minimum {window_size} values):',
        value=default_text
    )
    
else: # 'Use Live AAPL Data'
    st.markdown("---")
    st.success(f"Successfully loaded **{len(live_data_series)}** prices for the prediction window from Yahoo Finance. Status: {fetch_status}.")
    st.caption("The sequence used for prediction is the last `window_size` prices.")
    input_text = live_data_text # Use the fetched data as the input string


days = st.number_input('Days to predict', min_value=1, max_value=30, value=7)

if st.button('Predict'):
    try:
        # Parse the input text (whether live or manual)
        values = [float(x.strip()) for x in input_text.split(',') if x.strip()]

        if len(values) < window_size:
            st.error(f"Input must contain at least {window_size} prices. Please check the data source or your manual input.")
            st.stop()

        # Use only the last window_size values for the model input
        recent_values = pd.Series(values[-window_size:])
        
        # Generate prediction
        preds = predict_next_days(model, scaler, recent_values, days, window_size)

        # -----------------------------
        # Create date index for x-axis
        # -----------------------------
        # Use the data that was actually fed to the model (the last W values)
        input_plot_values = recent_values.tolist()
        
        # Calculate dates backward from today (or the last data point)
        # If using live data, history_dates should reflect the actual trading days/dates
        # For simplicity, we assume sequential days starting from the end of the data.
        
        # Determine the last date of the input data. We use the index if available, otherwise assume today - 1
        if data_source == 'Use Live AAPL Data' and not live_data_series.index.empty:
            last_input_date = live_data_series.index[-1].date()
        else:
            last_input_date = datetime.today().date() - timedelta(days=1)
        
        # Create dates for the input history (W days leading up to last_input_date)
        history_dates = [last_input_date - timedelta(days=window_size - i - 1) for i in range(window_size)]
        
        # Prediction dates start after the last history date
        pred_dates = [history_dates[-1] + timedelta(days=i + 1) for i in range(days)]

        # Combine dates for plotting connection (last input value to first prediction)
        pred_x = [history_dates[-1]] + pred_dates
        pred_y = [input_plot_values[-1]] + list(preds)

        # -----------------------------
        # Plotting
        # -----------------------------
        fig = go.Figure()

        # Input History (Only the W values used for the model)
        fig.add_trace(go.Scatter(
            x=history_dates,
            y=input_plot_values,
            name=f'Input History (W={window_size} prices)',
            line=dict(color='blue'),
            mode='lines+markers'
        ))

        # Predicted values (connected)
        fig.add_trace(go.Scatter(
            x=pred_x,
            y=pred_y,
            name=f'Predicted Forecast ({days} days)',
            mode='lines+markers',
            line=dict(color='red', dash='dot')
        ))

        fig.update_layout(
            title='AAPL Price Forecast using LSTM',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        st.plotly_chart(fig, use_container_width=True)

        # Display predicted values in a DataFrame
        preds_df = pd.DataFrame({'Predicted_Close': preds}, index=pred_dates)
        st.subheader(f'Forecasted Prices for the Next {days} Days')
        st.dataframe(preds_df.style.format("{:.2f}"))

    except Exception as e:
        # Catch any unexpected errors during prediction or plotting
        st.error(f'An unexpected error occurred during prediction: {e}')

st.markdown('---')

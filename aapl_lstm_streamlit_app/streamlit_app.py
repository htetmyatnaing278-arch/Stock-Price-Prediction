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
from scipy import stats # Import stats globally for clarity

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
    # Ensure input_values is a list of floats
    input_values = [float(v) for v in input_values]
    
    if len(input_values) < window_size:
        # Pad or handle insufficient input if needed, though the UI requires window_size inputs
        input_values = [input_values[0]] * (window_size - len(input_values)) + input_values
        
    scaled = scaler.transform(np.array(input_values).reshape(-1, 1)).flatten()
    preds_scaled = []

    # Use only the last window_size values for the initial sequence
    current_sequence = scaled[-window_size:] 

    for _ in range(n_days):
        # Reshape for LSTM input: (1, window_size, 1)
        seq = np.array(current_sequence).reshape(1, window_size, 1)
        pred = model.predict(seq, verbose=0).flatten()[0]
        preds_scaled.append(pred)
        
        # Update sequence for the next prediction (walk forward)
        current_sequence = np.append(current_sequence[1:], pred)

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
st.title('AAPL Close Price — LSTM Predictor')

model, scaler, window_size = load_components()
st.success(f'Model loaded successfully — required window_size = {window_size}')

# -----------------------------
# Manual Input
# -----------------------------
st.subheader('Input Historical Data for Prediction')
latest_price = get_latest_price()
# Generate window_size random values around the latest price for default input
default_values = [str(round(latest_price + random.uniform(-3, 3), 2)) for _ in range(window_size)]
manual_text = st.text_area(f'Enter {window_size} recent Close prices (comma-separated):', value=','.join(default_values))
days = st.number_input('Days to predict into the future (max 30)', min_value=1, max_value=30, value=7)

# -----------------------------
# Prediction and Plot
# -----------------------------
if st.button('Predict Future Prices'):
    try:
        # 1. Input Processing
        values = [float(x.strip()) for x in manual_text.split(',') if x.strip()]
        
        if len(values) < window_size:
            st.error(f'Please enter at least {window_size} values for the input sequence.')
            st.stop()
        
        # Use the last window_size elements for the prediction
        input_history = values[-window_size:]
        
        # ---------------------------------------------
        # 2. CALIBRATION RUN (To calculate RMSE CI)
        # ---------------------------------------------
        # For calibration, we need data to compare against. Assuming the first 60% 
        # of the input is history (X) and the remaining is actuals (Y) for error estimation.
        
        # We will use the first half of the input as history (X_calib) to predict the second half (Y_calib)
        CALIB_SIZE = min(window_size, 30) # Use 30 days for calibration
        
        # X_calib: The first (window_size - CALIB_SIZE) days of input
        X_calib = input_history[:-CALIB_SIZE] 
        # Y_calib (Actuals): The last CALIB_SIZE days of input
        Y_calib = input_history[-CALIB_SIZE:] 

        # Predict the Y_calib period using the X_calib history
        predicted_calib = predict_sequence(
            model, 
            scaler, 
            X_calib, 
            n_days=CALIB_SIZE, 
            window_size=window_size
        )
        
        # Calculate RMSE CI using calibration data
        residual = np.array(Y_calib) - np.array(predicted_calib)
        squared_error = residual ** 2
        mean_squared_error = squared_error.mean()
        
        ci_level = 0.95
        mse_ci_low, mse_ci_high = stats.t.interval(
            ci_level, 
            len(squared_error) - 1,
            loc=mean_squared_error,
            scale=stats.sem(squared_error)
        )
        
        rmse_ci_high = np.sqrt(mse_ci_high)
        
        # Display Calibration Info
        st.subheader('Calibration Analysis')
        st.info(
            f'Historical error (RMSE CI) calculated over {CALIB_SIZE} days: Margin of Error (95% CI High Bound) = **${rmse_ci_high:.2f}**'
        )

        # ---------------------------------------------
        # 3. FORWARD PREDICTION RUN
        # ---------------------------------------------
        # Use the full input_history to predict the next 'days'
        future_predictions = predict_sequence(
            model, 
            scaler, 
            input_history, 
            n_days=days, 
            window_size=window_size
        )

        # ---------------------------------------------
        # 4. Dates
        # ---------------------------------------------
        today = datetime.today()
        # Dates for the input history (window_size days ending yesterday)
        history_dates = [today - timedelta(days=window_size - i) for i in range(window_size)]
        
        # Dates for the future prediction (starting today + 1)
        future_dates = [today + timedelta(days=i + 1) for i in range(days)]

        # ---------------------------------------------
        # 5. Plot (History + Future Prediction + CI Band)
        # ---------------------------------------------
        st.subheader('Future Price Forecast with 95% Confidence Band')
        
        # Apply the RMSE CI margin to the future predictions
        future_low = future_predictions - rmse_ci_high
        future_high = future_predictions + rmse_ci_high
        
        fig = go.Figure()

        # Trace 1: Historical Input Data
        fig.add_trace(go.Scatter(
            x=history_dates, 
            y=input_history, 
            name='Historical Input', 
            line=dict(color='green', width=2)
        ))
        
        # Trace 2: Predicted Upper Bound (Hidden line for filling)
        fig.add_trace(go.Scatter(
            x=future_dates, 
            y=future_high, 
            name='95% CI Upper', 
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        # Trace 3: Predicted Lower Bound (Filled area)
        fig.add_trace(go.Scatter(
            x=future_dates, 
            y=future_low, 
            name='95% CI Range', 
            mode='lines',
            line=dict(width=0),
            fill='tonexty', # Fill the area between upper and lower bound
            fillcolor='rgba(135, 206, 250, 0.3)' # Light blue transparent fill
        ))

        # Trace 4: Point Prediction Line
        fig.add_trace(go.Scatter(
            x=future_dates, 
            y=future_predictions, 
            name='Point Prediction', 
            line=dict(color='skyblue', width=3, dash='dot')
        ))

        fig.update_layout(
            title=f'AAPL Price Forecast: Next {days} Days',
            xaxis_title='Date',
            yaxis_title='Predicted Close Price ($)',
            yaxis=dict(tickformat="$,.2f"),
            xaxis=dict(tickformat='%Y-%m-%d'),
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------------
        # 6. Table (Future Predictions)
        # ---------------------------------------------
        st.subheader(f'Detailed Forecast: Next {days} Trading Days')
        
        # Create the DataFrame for future predictions
        future_df = pd.DataFrame({
            'Predicted Price ($)': future_predictions,
            'CI 95% Lower Bound ($)': future_low,
            'CI 95% Upper Bound ($)': future_high
        }, index=future_dates)
        
        future_df.index.name = "Forecast Date"
        
        # Display the DataFrame with formatting
        st.dataframe(future_df.style.format("${:.2f}"))

    except Exception as e:
        st.error(f'An error occurred during prediction: {e}')
        st.warning('Please ensure your input contains valid, comma-separated numbers and the number of inputs matches the required window size.')

st.markdown('---')

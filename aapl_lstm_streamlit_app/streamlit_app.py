# ----------------------------- #
# STREAMLIT APP
# ----------------------------- #
import os, random, numpy as np, pandas as pd, joblib, streamlit as st
import plotly.graph_objs as go
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# Reproducibility
random_seed = 123
random.seed(random_seed)
np.random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(random_seed)

# Load model, scaler, window size
@st.cache_resource
def load_components():
    model = load_model('lstm_aapl_model.h5', compile=False)
    scaler = joblib.load('scaler.pkl')
    with open('window_size.txt') as f:
        window_size = int(f.read().strip())
    return model, scaler, window_size

model, scaler, window_size = load_components()
st.title("📈 AAPL Close Price — LSTM Predictor")
st.success(f'Model loaded successfully — window_size = {window_size}')

# Manual input
st.subheader("Manual Input")
default_values = [str(round(170 + random.uniform(-3, 3), 2)) for _ in range(window_size)]
manual_text = st.text_area(f'Enter {window_size} recent Close prices (comma-separated):', value=','.join(default_values))
days = st.number_input('Days to predict', min_value=1, max_value=30, value=7)

# Prediction logic
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

# Run prediction
if st.button("Predict"):
    try:
        values = [float(x.strip()) for x in manual_text.split(',') if x.strip()]
        predicted = predict_future(model, scaler, values, days, window_size)

        today = datetime.today()
        future_dates = [today + timedelta(days=i) for i in range(days)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=future_dates, y=predicted, name='Predicted', line=dict(color='red', dash='dot')))
        fig.add_trace(go.Scatter(x=[today - timedelta(days=i) for i in range(len(values))][::-1], y=values, name='Input History', line=dict(color='green')))
        fig.update_layout(title='AAPL Close Price Prediction', xaxis_title='Date', yaxis_title='Price ($)', yaxis=dict(tickformat="$,.2f"))
        st.plotly_chart(fig, use_container_width=True)

        df_preds = pd.DataFrame({'Date': future_dates, 'Predicted_Close ($)': predicted})
        st.subheader("Predicted Prices")
        st.dataframe(df_preds.set_index('Date').style.format("${:.2f}"))
    except Exception as e:
        st.error(f"Error during prediction: {e}")

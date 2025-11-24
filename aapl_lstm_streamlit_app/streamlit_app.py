import os
import random
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objs as go
import tensorflow as tf
from tensorflow.keras.models import load_model

random_seed = 123
random.seed(random_seed)
np.random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
try:
    tf.random.set_seed(random_seed)
except Exception:
    pass

# CHANGE: Default window_size now 7 (was 90)
@st.cache_resource
def load_saved_components(model_path: str = 'lstm_aapl_model.h5', scaler_path: str = 'scaler.pkl', window_path: str = 'window_size.txt'):
    """Load model, scaler and window size from disk. Cached to speed up app."""
    model = load_model(model_path, compile=False)
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    window_size = 7
    if os.path.exists(window_path):
        try:
            with open(window_path, 'r') as f:
                window_size = int(f.read().strip())
        except Exception:
            window_size = 7
    return model, scaler, window_size

def create_sequences(data_array, window_size):
    X = []
    for i in range(window_size, len(data_array)):
        X.append(data_array[i - window_size:i, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X

def predict_next_days(model, scaler, recent_values, days, window_size):
    """Predict recursively next `days` values given recent_values (raw, unscaled pandas Series)."""
    scaled = scaler.transform(recent_values.values.reshape(-1, 1))
    scaled_list = list(scaled.flatten())
    preds_scaled = []
    for _ in range(days):
        input_seq = np.array(scaled_list[-window_size:]).reshape(1, window_size, 1)
        p = model.predict(input_seq, verbose=0)
        preds_scaled.append(p.flatten()[0])
        scaled_list.append(p.flatten()[0])
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    return preds

st.set_page_config(page_title='AAPL Close Price Predictor', layout='wide')
st.title('AAPL Close Price — LSTM Predictor')
st.write('Upload a CSV with a `Date` column and a `Close` column (or just a `Close` column).')

model, scaler, window_size = load_saved_components(
    'aapl_lstm_streamlit_app/lstm_aapl_model.h5',
    'aapl_lstm_streamlit_app/scaler.pkl',
    'aapl_lstm_streamlit_app/window_size.txt'
)

uploaded_file = st.file_uploader('Upload CSV file (Date,Close). If none provided, use your own data in repo.', type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info('No file uploaded. You can still enter a small recent Close history manually for a quick demo.')
    df = None

if df is not None:
    cols = [c.lower() for c in df.columns]
    if 'date' in cols:
        df.columns = [c.strip() for c in df.columns]
        date_col = [c for c in df.columns if c.lower() == 'date'][0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        df.set_index(date_col, inplace=True)
    if 'close' not in [c.lower() for c in df.columns]:
        st.error('Uploaded CSV must contain a `Close` column (case-insensitive).')
    else:
        close_col = [c for c in df.columns if c.lower() == 'close'][0]
        series_close = df[close_col].astype(float)
        st.subheader('Preview')
        st.dataframe(df.tail(10))
        # CHANGE: Days to predict box min=7, max=14, initial=7
        days = st.number_input('Days to predict (7 to 14)', min_value=7, max_value=14, value=7)
        # CHANGE: Recent length minimum is window_size (now 7)
        recent_len = st.number_input('How many recent days from your history to use for recursive prediction', min_value=window_size, max_value=len(series_close), value=window_size + 6)
        recent_values = series_close[-recent_len:]
        if st.button('Run prediction'):
            with st.spinner('Predicting...'):
                preds = predict_next_days(model, scaler, recent_values, days, window_size)
                last_index = series_close.index[-1]
                if isinstance(last_index, (pd.Timestamp, pd.DatetimeIndex.dtype)):
                    pred_index = pd.date_range(start=last_index + pd.Timedelta(days=1), periods=days, freq='B')
                else:
                    pred_index = range(len(series_close), len(series_close) + days)
                preds_df = pd.DataFrame({'Predicted_Close': preds}, index=pred_index)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=series_close.index, y=series_close.values, name='Historical Close', mode='lines'))
                # CHANGE: Predicted line is now red
                fig.add_trace(go.Scatter(x=preds_df.index, y=preds_df['Predicted_Close'], name='Predicted Close', mode='lines+markers', line=dict(color='red')))
                fig.update_layout(title='Historical Close and Predicted Close', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig, use_container_width=True)
                st.subheader('Predicted values')
                st.dataframe(preds_df)
                st.success('Prediction complete.')
else:
    st.subheader('Enter Recent Close Prices')
    st.write(f"Provide at least {window_size} recent Close values, comma-separated (most recent last). More history may improve results.")
    demo_default_len = max(window_size, 7)
    demo_values = [str(random.randint(80, 170)) for _ in range(demo_default_len)]
    manual_text = st.text_area(
        f'Enter last N days Close prices (min {window_size}), separated by commas.',
        value=','.join(demo_values)
    )
    try:
        values = [float(x.strip()) for x in manual_text.split(',') if x.strip() != '']
        valid_len = len(values) >= window_size
    except Exception:
        values = []
        valid_len = False
    days = st.number_input(
        'Days to predict (demo)', min_value=7, max_value=14, value=7, key='demo_days'
    )
    if st.button('Run demo prediction'):
        if not valid_len:
            st.error(f'Provide at least {window_size} values.')
        else:
            recent_values = pd.Series(values)
            preds = predict_next_days(model, scaler, recent_values, days, window_size)
            pred_index = list(range(len(values), len(values) + days))
            preds_df = pd.DataFrame({'Predicted_Close': preds}, index=pred_index)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(values))), y=values, name=f'Manual input'))
            fig.add_trace(go.Scatter(x=preds_df.index, y=preds_df['Predicted_Close'], name='Predicted', line=dict(color='red')))
            fig.update_layout(title='Manual input — Predicted Close')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(preds_df)

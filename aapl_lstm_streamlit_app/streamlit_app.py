
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


@st.cache_resource
def load_saved_components(
    model_path: str = 'aapl_lstm_streamlit_app/lstm_aapl_model.h5',
    scaler_path: str = 'aapl_lstm_streamlit_app/scaler.pkl',
    window_path: str = 'aapl_lstm_streamlit_app/window_size.txt'
):

    """Load model, scaler and window size from disk. Cached to speed up app."""

    model = load_model(model_path, compile=False)


    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)


    window_size = 90
    if os.path.exists(window_path):
        try:
            with open(window_path, 'r') as f:
                window_size = int(f.read().strip())
        except Exception:
            window_size = 90

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


st.sidebar.header('Model files (repo)')
model_path = st.sidebar.text_input('Saved model path', 'aapl_lstm_streamlit_app/lstm_aapl_model.h5')
scaler_path = st.sidebar.text_input('Saved scaler path', 'aapl_lstm_streamlit_app/scaler.pkl')
window_path = st.sidebar.text_input('Window size file path', 'aapl_lstm_streamlit_app/window_size.txt')

model, scaler, window_size = load_saved_components(model_path, scaler_path, window_path)
st.sidebar.success(f'Model loaded — window_size = {window_size}')


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


        days = st.number_input('Days to predict', min_value=1, max_value=365, value=7)


        recent_len = st.number_input('How many recent days from your history to use for recursive prediction', min_value=window_size, max_value=len(series_close), value=window_size + 20)

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
                fig.add_trace(go.Scatter(x=preds_df.index, y=preds_df['Predicted_Close'], name='Predicted Close', mode='lines+markers'))
                fig.update_layout(title='Historical Close and Predicted Close', xaxis_title='Date', yaxis_title='Price')
                st.plotly_chart(fig, use_container_width=True)

                st.subheader('Predicted values')
                st.dataframe(preds_df)


                st.success('Prediction complete.')

else:

    st.subheader('Quick demo (manual input)')
    manual_text = st.text_area('Enter recent Close prices as comma-separated numbers (most recent last). Need at least window_size values.', value=','.join(['150'] * (window_size + 10)))
    if st.button('Run demo prediction'):
        try:
            values = [float(x.strip()) for x in manual_text.split(',') if x.strip()!='']
            if len(values) < window_size:
                st.error(f'Provide at least {window_size} values.')
            else:
                recent_values = pd.Series(values)
                days = st.number_input('Days to predict (demo)', min_value=1, max_value=365, value=7, key='demo_days')
                preds = predict_next_days(model, scaler, recent_values, days, window_size)

                pred_index = list(range(len(values), len(values) + days))
                preds_df = pd.DataFrame({'Predicted_Close': preds}, index=pred_index)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(len(values))), y=values, name='Manual history'))
                fig.add_trace(go.Scatter(x=preds_df.index, y=preds_df['Predicted_Close'], name='Predicted'))
                fig.update_layout(title='Manual input — Predicted Close')
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(preds_df)
        except Exception as e:
            st.error(f'Error parsing input: {e}')

st.markdown('---')
st.caption('Note: This app expects the model to have been trained and saved separately. Do not use predictions for trading without further validation.')


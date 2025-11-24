# streamlit_app.py
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
from sklearn.metrics import mean_absolute_percentage_error
from scipy import stats
from scipy.stats import normaltest

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
# Utility: RMSE CI (95%)
# -----------------------------
def compute_rmse_ci(residuals, ci=0.95):
    """
    Compute 95% CI for RMSE by:
      1) computing squared errors
      2) CI for mean squared error (t-interval)
      3) sqrt the interval to get RMSE CI (clamp low>=0)
    residuals: numpy array (y_true - y_pred)
    returns (rmse_low, rmse_high)
    """
    residuals = np.array(residuals).astype(float).flatten()
    if len(residuals) < 2:
        return (np.nan, np.nan)
    squared_error = np.square(residuals)
    mse_mean = np.mean(squared_error)
    sem = stats.sem(squared_error)
    df = len(squared_error) - 1

    # t interval for MSE mean
    try:
        mse_ci_low, mse_ci_high = stats.t.interval(ci, df, loc=mse_mean, scale=sem)
    except Exception:
        # fallback to normal approximation if t fails
        z = stats.norm.ppf(0.5 + ci/2)
        mse_ci_low = mse_mean - z * sem
        mse_ci_high = mse_mean + z * sem

    mse_ci_low = max(mse_ci_low, 0.0)
    rmse_low = np.sqrt(mse_ci_low)
    rmse_high = np.sqrt(max(mse_ci_high, 0.0))
    return (rmse_low, rmse_high)

# -----------------------------
# Load model, scaler, and window size
# -----------------------------
@st.cache_resource
def load_saved_components():
    model_path = 'aapl_lstm_streamlit_app/lstm_aapl_model.h5'
    scaler_path = 'aapl_lstm_streamlit_app/scaler.pkl'
    window_path = 'aapl_lstm_streamlit_app/window_size.txt'

    # helpful errors if missing
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
# Prediction helper (multi-step) - price-model style
# -----------------------------
def predict_next_days(model, scaler, recent_values, days, window_size):
    """
    recent_values: pd.Series or 1D array of raw prices (length >= window_size)
    days: number of future days to predict
    window_size: model's window size
    returns: numpy array of predicted raw prices length=days
    """
    recent_values = pd.Series(recent_values).reset_index(drop=True)

    # ensure we have at least window_size points; if not pad with last value
    if len(recent_values) < window_size:
        pad_len = window_size - len(recent_values)
        pad_vals = [recent_values.iloc[0]] * pad_len
        recent_values = pd.Series(pad_vals + recent_values.tolist())

    # use last window_size for initial scaled sequence
    initial_slice = recent_values.tail(window_size).values.reshape(-1, 1)
    scaled = scaler.transform(initial_slice)
    scaled_list = list(scaled.flatten())

    preds_scaled = []
    for _ in range(days):
        seq = np.array(scaled_list[-window_size:]).reshape(1, window_size, 1)
        p = model.predict(seq, verbose=0)
        p_val = float(p.flatten()[0])
        preds_scaled.append(p_val)
        scaled_list.append(p_val)

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    return preds

# -----------------------------
# Get historical AAPL close prices (for building test set)
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_aapl_history(period='3y'):
    """
    Fetches historical AAPL daily close prices for the given period.
    Default period='3y' gives enough data for training/test split.
    """
    try:
        df_hist = yf.download("AAPL", period=period, interval="1d", progress=False)
        if df_hist is None or df_hist.empty:
            raise ValueError("Empty history from yfinance")
        close = df_hist['Close'].reset_index(drop=True)
        return close
    except Exception as e:
        st.warning(f"Failed to fetch historical data from Yahoo Finance: {e}")
        return pd.Series(dtype=float)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title='AAPL Close Price Predictor', layout='wide')
st.title('AAPL Close Price — LSTM Predictor (with Evaluation & CI)')

# load model/scaler/window
model, scaler, window_size = load_saved_components()
st.success(f'Model loaded — window_size = {window_size}')

# -----------------------------
# Section: Evaluation using historical data (train/test comparable to your training snippet)
# -----------------------------
st.header("Automatic Evaluation (uses historical AAPL data)")

# fetch history
close_series = fetch_aapl_history(period='3y')  # 3 years of data
if close_series.empty or len(close_series) < (window_size + 10):
    st.warning("Not enough historical data to perform automatic evaluation. Manual prediction below will still work.")
else:
    # build train/test split 80/20 as in your training snippet
    N = len(close_series)
    train_size = int(N * 0.8)
    train_close = close_series[:train_size]
    test_close = close_series[train_size:]

    # scale using loaded scaler (assumes scaler was fit on training prices during model creation)
    try:
        train_scaled = scaler.transform(train_close.values.reshape(-1, 1))
        test_scaled = scaler.transform(test_close.values.reshape(-1, 1))
    except Exception as e:
        st.warning(f"Scaler transform failed: {e}. Attempting to fit a new scaler locally for evaluation.")
        from sklearn.preprocessing import MinMaxScaler
        tmp_scaler = MinMaxScaler()
        train_scaled = tmp_scaler.fit_transform(train_close.values.reshape(-1, 1))
        test_scaled = tmp_scaler.transform(test_close.values.reshape(-1, 1))
        # override scaler for inverse later in evaluation
        eval_scaler = tmp_scaler
    else:
        eval_scaler = scaler

    # create sequences for X_test/Y_test using window_size
    X_test, y_test = [], []
    for i in range(window_size, len(test_scaled)):
        X_test.append(test_scaled[i - window_size: i, 0])
        y_test.append(test_scaled[i, 0])

    if len(X_test) == 0:
        st.warning("After sequence creation, no test samples available (not enough data).")
    else:
        X_test = np.array(X_test)
        y_test = np.array(y_test).reshape(-1, 1)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Evaluate model on X_test
        try:
            result = model.evaluate(X_test, y_test, verbose=0)
            prediction_scaled = model.predict(X_test, verbose=0)
        except Exception as e:
            st.error(f"Model evaluation failed: {e}")
            prediction_scaled = None
            result = None

        if prediction_scaled is not None:
            # inverse transform to prices
            y_test_true = eval_scaler.inverse_transform(y_test).flatten()
            y_test_prediction = eval_scaler.inverse_transform(prediction_scaled).flatten()

            # metrics
            MAPE = mean_absolute_percentage_error(y_test_true, y_test_prediction)
            accuracy = 1 - MAPE

            st.subheader("Evaluation Results")
            if result is not None:
                loss_val = result[0] if isinstance(result, (list, tuple)) else result
                st.write(f"Loss (on test): {loss_val}")
            st.write(f"MAPE: {MAPE:.6f}")
            st.write(f"Accuracy (1 - MAPE): {accuracy:.6f}")
            st.write(f"Number of test samples: {len(y_test_true)}")

            # residuals and normality test
            residual = y_test_true - y_test_prediction
            stat, p = normaltest(residual)
            st.write("Normality test (D'Agostino): statistic = {:.4f}, p = {:.4f}".format(stat, p))
            if p > 0.05:
                st.info("Residuals pass normality test (p > 0.05) — cannot reject normality.")
            else:
                st.info("Residuals fail normality test (p <= 0.05) — deviate from normality.")

            # RMSE CI
            rmse_low, rmse_high = compute_rmse_ci(residual, ci=0.95)
            st.write(f"95% RMSE CI: ({rmse_low:.6f}, {rmse_high:.6f})")

            # Show a small comparison table
            compare_df = pd.DataFrame({
                'y_true': y_test_true,
                'y_pred': y_test_prediction,
                'residual': residual
            })
            st.subheader("First 10 test predictions vs actual")
            st.dataframe(compare_df.head(10).style.format("${:,.2f}"))

            # Plot a slice (first 200 points or all if less) for visual comparison
            take_n = min(200, len(y_test_true))
            times = pd.date_range(end=datetime.today().date(), periods=len(y_test_true)).tolist()
            plot_dates = times[-take_n:]

            fig_eval = go.Figure()
            fig_eval.add_trace(go.Scatter(
                x=plot_dates,
                y=y_test_true[-take_n:],
                name='Actual (test)',
                line=dict(color='red')
            ))
            fig_eval.add_trace(go.Scatter(
                x=plot_dates,
                y=y_test_prediction[-take_n:],
                name='Predicted (test)',
                line=dict(color='skyblue')
            ))
            fig_eval.update_layout(
                title="Test set: Actual vs Predicted (sample)",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                yaxis=dict(tickformat="$,.2f")
            )
            st.plotly_chart(fig_eval, use_container_width=True)

# -----------------------------
# Manual Input Prediction (unchanged)
# -----------------------------
st.header("Manual Input Prediction")

# Get latest price for default history generation (try live)
def get_latest_aapl_price():
    try:
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="5d", interval="1d", progress=False)
        if hist is None or hist.empty:
            raise ValueError("empty history")
        return float(hist['Close'].iloc[-1])
    except Exception:
        return None

latest_price = get_latest_aapl_price()
if latest_price is None:
    st.warning("Could not fetch live AAPL price; using fallback 275.00 for defaults.")
    latest_price = 275.0

st.info(f"Latest AAPL close (used to seed defaults): ${latest_price:.2f}")

# manual input area
manual_count = window_size
default_values = [str(round(latest_price + random.uniform(-3, 3), 2)) for _ in range(manual_count)]
manual_text = st.text_area(f"Enter {manual_count} recent Close prices (comma-separated):", value=','.join(default_values))
days = st.number_input('Days to predict (manual):', min_value=1, max_value=30, value=7)

if st.button('Predict (manual)'):
    try:
        values = [float(x.strip()) for x in manual_text.split(',') if x.strip()]
        if len(values) < window_size:
            repeats = (window_size // len(values)) + 1
            values = (values * repeats)[:window_size]

        recent_values = pd.Series(values[-window_size:])
        preds = predict_next_days(model, scaler, recent_values, days, window_size)

        # plotting dates
        today = datetime.today()
        history_dates = [today - timedelta(days=len(values) - i - 1) for i in range(len(values))]
        pred_dates = [history_dates[-1] + timedelta(days=i + 1) for i in range(days)]

        # plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_dates,
            y=values,
            name='Manual history',
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=[history_dates[-1]] + pred_dates,
            y=[values[-1]] + list(preds),
            name=f'Predicted ({days} days)',
            mode='lines+markers',
            line=dict(color='red')
        ))
        fig.update_layout(
            title='Manual Input — Predicted Close',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            yaxis=dict(tickformat="$,.2f"),
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        st.plotly_chart(fig, use_container_width=True)

        preds_df = pd.DataFrame({'Predicted_Close ($)': preds}, index=pred_dates)
        st.subheader(f'Forecast for the Next {days} Days')
        st.dataframe(preds_df.style.format("${:.2f}"))

    except Exception as e:
        st.error(f'Input error: {e}')

st.markdown('---')

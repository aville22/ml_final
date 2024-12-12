import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import yfinance as yf
from datetime import date, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import streamlit as st
import warnings

# Constants
close = "Close"
days = 30      # Number of days to predict
n = 20         # LSTM input sequence length
epochs = 50    # Number of epochs for LSTM
learning_rate = 0.001
batch_size = 64
number_nodes = 128
lag = 40       # Lags for ACF/PACF

#------------------------------------------------
# Utility Functions
#------------------------------------------------
def data_allocation(df, days):
    train_len_val = len(df) - days
    train, test = df[close].iloc[0:train_len_val], df[close].iloc[train_len_val:]
    return train, test

def apply_transform(data, n: int):
    middle_data = []
    target_data = []
    for i in range(n, len(data)):
        input_sequence = data[i-n:i]  
        middle_data.append(input_sequence) 
        target_data.append(data[i])
    middle_data = np.array(middle_data).reshape((len(middle_data), n, 1))
    target_data = np.array(target_data)
    return middle_data, target_data

def calculate_accuracy(true_values, predictions):
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    return mse, rmse, mae

def Error_Evaluation(train_data, predict_train_data, n):
    # errors = actual - predicted (on train set)
    errors = []
    actual_data = train_data.values
    for i in range(len(predict_train_data)):
        err = actual_data[n + i] - predict_train_data[i]
        errors.append(err)
    return errors

def Parameter_calculation(data, p_max=3, d_max=3, q_max=3):
    # Brute force (p,d,q) to find best AIC
    warnings.filterwarnings("ignore")
    best_aic = float("inf")
    best_order = None
    for p in range(p_max):
        for d in range(d_max):
            for q in range(q_max):
                try:
                    model = ARIMA(data, order=(p,d,q))
                    res = model.fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p,d,q)
                except:
                    continue
    if best_order is None:
        best_order = (1,1,1)
    return best_order

def ARIMA_Model(train, len_test, ord):
    model = ARIMA(train, order=ord)
    model = model.fit()
    predictions = model.predict(start=len(train), end=len(train) + len_test, typ='levels')
    full_predictions = model.predict(start=0, end=len(train)-1, typ='levels')
    return model, predictions, full_predictions

def Final_Predictions(predictions_errors, predictions, days):
    final_values = []
    for i in range(days):
        final_values.append(predictions_errors[i] + predictions[i])
    return final_values

def plot_time_series(x, y, title, xlabel='Date', ylabel='Close Price', legend_labels=['Actual', 'Predicted'], color='red'):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(x.index, x, label=legend_labels[0])
    ax.plot(x.index, y, label=legend_labels[1], color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig

def plot_series(df, title='Raw Time Series'):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df["Date"], df[close], label='Close Price')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    return fig

def plot_train_test(train, test):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(train.index, train, label='Train')
    ax.plot(test.index, test, label='Test', color='orange')
    ax.set_title('Train and Test Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    return fig

def plot_errors(errors, title='Prediction Errors'):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(errors, label='Errors')
    ax.set_title(title)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Error')
    ax.legend()
    return fig

def plot_final(test, final_predictions, title='Final Predictions with Error Correction'):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(test.index, test, label='Actual')
    ax.plot(test.index, final_predictions, label='Corrected Prediction', color='green')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    return fig

def plot_accuracy_bar(mse, rmse, mae, title="Model Accuracy Metrics"):
    fig, ax = plt.subplots(figsize=(6,4))
    metrics = ['MSE', 'RMSE', 'MAE']
    values = [mse, rmse, mae]
    ax.bar(metrics, values, color=['blue', 'orange', 'green'])
    ax.set_title(title)
    return fig

def LSTM_model(train, n, number_nodes, learning_rate, epochs, batch_size):
    middle_data, target_data = apply_transform(train.values, n)
    model = tf.keras.Sequential([
        tf.keras.layers.Input((n,1)),
        tf.keras.layers.LSTM(number_nodes, input_shape=(n, 1)),
        tf.keras.layers.Dense(units = number_nodes, activation = "relu"),
        tf.keras.layers.Dense(units = number_nodes, activation = "relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=["mean_absolute_error"])
    model.fit(middle_data, target_data, epochs=epochs, batch_size=batch_size, verbose=0)
    full_predictions = model.predict(middle_data).flatten()
    return model, full_predictions

def forecast_lstm(model, train, test, n, days):
    last_sequence = train[-n:].values.reshape((1, n, 1))
    predictions = []
    for i in range(days+1):
        next_prediction = model.predict(last_sequence).flatten()[0]
        predictions.append(next_prediction)
        if i < len(test):
            actual_value = float(test.iloc[i])
            new_row = np.concatenate([last_sequence[:, 1:, :], np.array([[[actual_value]]])], axis=1)
        else:
            new_row = np.concatenate([last_sequence[:, 1:, :], np.array([[[next_prediction]]])], axis=1)
        last_sequence = new_row.reshape((1, n, 1))
    return predictions

#------------------------------------------------
# Streamlit App
#------------------------------------------------
def main():
    st.title("Stock Price Forecasting (LSTM + ARIMA on Errors)")
    st.markdown("""
    This app forecasts stock prices using:
    - An LSTM model trained on historical data.
    - An ARIMA model fitted on the LSTM training errors to correct future predictions.
    """)

    ticker = st.text_input("Enter stock ticker", "GOOG")
    years = st.number_input("Number of years of data", min_value=1, max_value=50, value=20)
    run_button = st.button("Run Forecast")

    if run_button:
        # Data loading
        st.subheader("Loading and Preparing Data...")
        today = date.today()
        end_date = today.strftime("%Y-%m-%d")
        d1 = date.today() - timedelta(days=365 * years)
        start_date = d1.strftime("%Y-%m-%d")

        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.error("No data retrieved. Try another ticker or date range.")
            return

        data["Date"] = data.index
        df = data[["Date", "Close"]].reset_index(drop=True)

        st.write("Data Head:")
        st.dataframe(df.head())
        fig_raw = plot_series(df, title=f"{ticker} Raw Time Series Data")
        st.pyplot(fig_raw)

        train, test = data_allocation(df, days)
        fig_tt = plot_train_test(train, test)
        st.pyplot(fig_tt)

        st.write(f"Using LSTM lag (n): {n}, and days to predict: {days}")
        st.subheader("Training LSTM...")
        model, full_predictions = LSTM_model(train, n, number_nodes, learning_rate, epochs, batch_size)

        # Plot LSTM predictions on train
        fig_lstm_train = plot_time_series(train[n:], full_predictions, "LSTM Predictions vs Actual (Train)")
        st.pyplot(fig_lstm_train)

        # LSTM forecast on test
        lstm_predictions = forecast_lstm(model, train, test, n, days)
        fig_lstm_test = plot_time_series(test, lstm_predictions[:-1], "LSTM Predictions vs Actual (Test)", 
                                         legend_labels=['Actual','Predicted'])
        st.pyplot(fig_lstm_test)

        # Errors from training
        errors_data = Error_Evaluation(train, full_predictions, n)
        fig_errors = plot_errors(errors_data, title='LSTM Training Errors')
        st.pyplot(fig_errors)

        st.subheader(f"{days}-Day Prediction Values from LSTM")
        lstm_table = pd.DataFrame({
            "Day": np.arange(1, days+1),
            "Actual": [test.iloc[i] if i < len(test) else None for i in range(days)],
            "LSTM_Pred": lstm_predictions[:days]
        })
        st.dataframe(lstm_table)

        lstm_mse, lstm_rmse, lstm_mae = calculate_accuracy(test[:days], lstm_predictions[:days])
        st.subheader("LSTM Model Accuracy")
        st.write(f"MSE: {lstm_mse}, RMSE: {lstm_rmse}, MAE: {lstm_mae}")
        fig_lstm_acc = plot_accuracy_bar(lstm_mse, lstm_rmse, lstm_mae, "LSTM Model Accuracy")
        st.pyplot(fig_lstm_acc)

        st.subheader("Fitting ARIMA on LSTM Errors")
        ord = Parameter_calculation(errors_data)
        # Show chosen order
        st.write(f"Selected ARIMA order: {ord}")

        Arima_Model, predictions_errors, full_predictions_errors = ARIMA_Model(errors_data, len(test), ord)

        st.subheader(f"ARIMA Model {days}-Day Predictions of Errors")
        arima_table = pd.DataFrame({
            "Day": np.arange(1, len(predictions_errors)+1),
            "Predicted_Error": predictions_errors
        })
        st.dataframe(arima_table)

        arima_mse, arima_rmse, arima_mae = calculate_accuracy(errors_data, full_predictions_errors)
        st.subheader("ARIMA Model (Errors) Accuracy on Training Errors")
        st.write(f"MSE: {arima_mse}, RMSE: {arima_rmse}, MAE: {arima_mae}")
        fig_arima_acc = plot_accuracy_bar(arima_mse, arima_rmse, arima_mae, "ARIMA Model (Errors) Accuracy")
        st.pyplot(fig_arima_acc)

        st.subheader("ARIMA Model Summary")
        st.text(str(Arima_Model.summary()))

        final_predictions = Final_Predictions(predictions_errors, lstm_predictions, days)
        fig_final_preds = plot_final(test[:days], final_predictions[:days])
        st.pyplot(fig_final_preds)

        st.subheader("Final (LSTM + ARIMA) Predictions")
        final_table = pd.DataFrame({
            "Day": np.arange(1, days+1),
            "Actual": [test.iloc[i] if i < len(test) else None for i in range(days)],
            "LSTM_Pred": lstm_predictions[:days],
            "Final_Pred": final_predictions[:days]
        })
        st.dataframe(final_table)

        final_mse, final_rmse, final_mae = calculate_accuracy(test[:days], final_predictions[:days])
        st.subheader("Final Model (LSTM+ARIMA) Accuracy")
        st.write(f"MSE: {final_mse}, RMSE: {final_rmse}, MAE: {final_mae}")
        fig_final_acc = plot_accuracy_bar(final_mse, final_rmse, final_mae, "Final Model Accuracy")
        st.pyplot(fig_final_acc)

        # Next data point forecast
        if len(lstm_predictions) > days and len(predictions_errors) > days:
            next_point_forecast = lstm_predictions[days] + predictions_errors[days]
            st.subheader("Forecast Value of Next Data Point (beyond test window)")
            st.write(next_point_forecast)
        else:
            st.write("Not enough data to forecast the next data point beyond the test period.")

        st.success("Forecasting completed!")

if __name__ == "__main__":
    # Set page layout
    st.set_page_config(layout="wide")
    main()

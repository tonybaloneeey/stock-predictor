import os
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Activation
from keras import optimizers
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from pandas.tseries.offsets import BDay
from datetime import datetime
import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st

pd.set_option('display.max_columns', None)

def train(stock, recent_days=30, epochs=50, batch_size=10, validation_split=0.1, custom=False):
    # st.write('Training model for', stock)

    data = yf.download(tickers=stock, start='2020-01-01', end='2023-08-03')

    data['TargetNextClose'] = data['Adj Close'].shift(-1)

    data.dropna(inplace=True)
    data.reset_index(inplace=True)
    data.drop(['Date', 'Close', 'Volume'], axis=1, inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X = []
    for j in range(data.shape[1] - 2):
        X.append([])
        for i in range(recent_days, scaled_data.shape[0]):
            X[j].append(scaled_data[i - recent_days:i, j])

    X = np.moveaxis(X, [0], [2])

    X, yi = np.array(X), np.array(scaled_data[recent_days:, -1])

    y = np.reshape(yi, (len(yi), 1))

    train_percentage = 0.8

    split_limit = int(len(X) * train_percentage)
    X_train, X_test = X[:split_limit], X[split_limit:]
    y_train, y_test = y[:split_limit], y[split_limit:]
    np.random.seed(10)

    lstm_input = Input(shape=(recent_days, data.shape[1] - 2), name='lstm_input')
    inputs = LSTM(150, name='first_layer')(lstm_input)
    inputs = Dense(1, name='dense_layer')(inputs)
    output = Activation('linear', name='output')(inputs)
    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs,
              shuffle=True, validation_split=validation_split)

    y_pred = model.predict(X_test)

    mm_scaler_pred = MinMaxScaler()
    mm_scaler_pred.min_, mm_scaler_pred.scale_ = scaler.min_[0], scaler.scale_[0]

    y_test_scaled = mm_scaler_pred.inverse_transform(y_test)
    y_pred_scaled = mm_scaler_pred.inverse_transform(y_pred)

    st.write('Mean Average Error: ', str(round(mean_absolute_error(y_test, y_pred), 3) * 100) + "%")

    plt.figure(figsize=(30, 8))
    plt.plot(y_test_scaled, color='black', label='test')
    plt.plot(y_pred_scaled, color='green', label='pred')
    plt.legend()
    st.pyplot(plt)

    if not custom:
        model.save("models/" + stock + '_model.keras')
    else:
        model.save("models/" + stock + '_' + str(recent_days) + '_custom_model.keras')


def predict(stock, days=30, custom=False):

    predict_ticker = stock

    if not custom:
        model = load_model("models/" + predict_ticker + "_model.keras")
    else:
        model = load_model("models/" + predict_ticker + "_custom_model.keras")
        days = int(stock[-2:])
        stock = stock[:-3]

    today = datetime.today()
    date_behind = today - (days + 10) * BDay()
    data = yf.download(tickers=stock, start=date_behind, end=today)

    data['TargetNextClose'] = data['Adj Close'].shift(-1)

    data.reset_index(inplace=True)
    data.drop(['Date', 'Volume', 'Close'], axis=1, inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X = []
    recent_days = days
    for j in range(data.shape[1] - 2):
        X.append([])
        for i in range(recent_days, scaled_data.shape[0]):
            X[j].append(scaled_data[i - recent_days:i, j])

    X = np.moveaxis(X, [0], [2])
    X = np.array(X)

    pred = model.predict(X)

    mm_scaler_pred = MinMaxScaler()
    mm_scaler_pred.min_, mm_scaler_pred.scale_ = scaler.min_[0], scaler.scale_[0]
    pred_scaled = mm_scaler_pred.inverse_transform(pred)

    return pred_scaled[-1][0]


##########################################################################################

st.title('Stock Price Prediction')

pretrained_models = ['TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOG']
custom_models = []

for filename in os.listdir('models'):
    if '_custom_model.keras' in filename:
        ticker = filename.split('_custom_model.keras')[0]
        custom_models.append(ticker)

col1, col2 = st.columns(2)

with col1:
    model_list = st.radio('Choose a pre-trained model:', list(pretrained_models))
    if model_list:
        predicted_price = predict(model_list)
        st.info('Predicted close price for ' + model_list + ' tomorrow is $' + str(round(predicted_price, 2)))
    custom_model_list = st.radio('Choose a custom model:', list(custom_models))
    if custom_model_list:
        predicted_price = predict(custom_model_list, custom=True)
        st.info('Predicted close price for ' + custom_model_list + ' tomorrow is $' + str(round(predicted_price, 2)))
with col2:
    with st.form("custom_ticker"):
        model_input = st.text_input('Enter ticker')
        train_button = st.form_submit_button(label="Train Model")
        slider_recent_days = st.slider("Recent Days", min_value=5, max_value=100, value=30)
        slider_epochs = st.slider("Epochs", min_value=1, max_value=100, value=50)
        slider_batch_size = st.slider("Batch Size", min_value=1, max_value=100, value=10)
        slider_validation_split = st.slider("Validation Split", min_value=0.05, max_value=0.5, value=0.1)
    if train_button and model_input and model_input in custom_models:
        st.write("Model for", model_input, "already exists!")
    if train_button and model_input and model_input not in custom_models:
        st.write("TRAINING " + model_input + " STOCK MODEL...")
        train(model_input, recent_days=slider_recent_days, epochs=slider_epochs, batch_size=slider_batch_size,
              validation_split=slider_validation_split, custom=True)

        st.write("Custom model for", model_input, "has been trained successfully!")

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pandas_ta as ta
import numpy as np

predict_ticker = 'TSLA'

model = load_model(predict_ticker + "_model.keras")

data = yf.download(tickers='TSLA', start='2021-08-02', end='2023-08-02')

data['RSI'] = ta.rsi(data.Close, length=15)
data['EMAF'] = ta.ema(data.Close, length=20)
data['EMAM'] = ta.ema(data.Close, length=100)
data['EMAS'] = ta.ema(data.Close, length=150)

data['Target'] = data['Adj Close'] - data.Open
data['Target'] = data['Target'].shift(-1)

data['TargetNextClose'] = data['Adj Close'].shift(-1)

data.dropna(inplace=True)
data.reset_index(inplace=True)
data.drop(['Date', 'Volume', 'Close'], axis=1, inplace=True)

data_set = data.iloc[:, 0:11]

scaler = MinMaxScaler(feature_range=(0, 1))
data_set_scaled = scaler.fit_transform(data_set)

X = []
recent_days = 30
for j in range(8):
    X.append([])
    for i in range(recent_days, data_set_scaled.shape[0]):
        X[j].append(data_set_scaled[i - recent_days:i, j])

X = np.moveaxis(X, [0], [2])

X = np.array(X)

pred = model.predict(X)

mm_scaler_pred = MinMaxScaler()
mm_scaler_pred.min_, mm_scaler_pred.scale_ = scaler.min_[0], scaler.scale_[0]

pred_scaled = mm_scaler_pred.inverse_transform(pred)

print(pred_scaled[-1])


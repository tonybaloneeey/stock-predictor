from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import mean_absolute_error
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Activation
from keras import optimizers
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_ta as ta
import numpy as np

data = yf.download(tickers='AMZN', start='2010-01-01', end='2023-01-01')

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
data_set.head(20)

scaler = MinMaxScaler(feature_range=(0, 1))
data_set_scaled = scaler.fit_transform(data_set)

X = []
recent_days = 30
for j in range(8):
    X.append([])
    for i in range(recent_days, data_set_scaled.shape[0]):
        X[j].append(data_set_scaled[i - recent_days:i, j])

X = np.moveaxis(X, [0], [2])

X, yi = np.array(X), np.array(data_set_scaled[recent_days:, -1])
y = np.reshape(yi, (len(yi), 1))

split_limit = int(len(X) * 0.8)
X_train, X_test = X[:split_limit], X[split_limit:]
y_train, y_test = y[:split_limit], y[split_limit:]
np.random.seed(10)

lstm_input = Input(shape=(recent_days, 8), name='lstm_input')
inputs = LSTM(150, name='first_layer')(lstm_input)
inputs = Dense(1, name='dense_layer')(inputs)
output = Activation('linear', name='output')(inputs)
model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam()
model.compile(optimizer=adam, loss='mse')
model.fit(x=X_train, y=y_train, batch_size=10, epochs=50, shuffle=True, validation_split = 0.1)

y_pred = model.predict(X_test)
print('MAE: ', mean_absolute_error(y_test, y_pred))

plt.figure(figsize=(30, 8))
plt.plot(y_test, color='black', label='test')
plt.plot(y_pred, color='green', label='pred')
plt.legend()
plt.show()

model.save('trained_model.keras')
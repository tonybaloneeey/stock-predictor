from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Activation
from keras import optimizers
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_ta as ta
import numpy as np

ticker = 'TSLA'

data = yf.download(tickers=ticker, start='2015-01-01', end='2023-08-03')

data['RSI'] = ta.rsi(data.Close, length=15)
data['EMAF'] = ta.ema(data.Close, length=20)
data['EMAM'] = ta.ema(data.Close, length=100)
data['EMAS'] = ta.ema(data.Close, length=150)

data['Target'] = data['Adj Close'] - data.Open
data['Target'] = data['Target'].shift(-1)

data['TargetNextClose'] = data['Adj Close'].shift(-1)

data.dropna(inplace=True)
data.reset_index(inplace=True)
data.drop(['Date', 'Close', 'Volume'], axis=1, inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

X = []
recent_days = 30
for j in range(data.shape[1] - 2):
    X.append([])
    for i in range(recent_days, scaled_data.shape[0]):
        X[j].append(scaled_data[i - recent_days:i, j])

X = np.moveaxis(X, [0], [2])

# Convert X to numpy array and yi to numpy array
X, yi = np.array(X), np.array(scaled_data[recent_days:, -1])

# Convert yi to array of [num_samples, num_output_features], and model only has 1 output feature
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
model.fit(x=X_train, y=y_train, batch_size=10, epochs=50, shuffle=True, validation_split=0.1)

y_pred = model.predict(X_test)

mm_scaler_pred = MinMaxScaler()
mm_scaler_pred.min_, mm_scaler_pred.scale_ = scaler.min_[0], scaler.scale_[0]

y_test_scaled = mm_scaler_pred.inverse_transform(y_test)
y_pred_scaled = mm_scaler_pred.inverse_transform(y_pred)

print('MAE: ', mean_absolute_error(y_test, y_pred))

plt.figure(figsize=(30, 8))
plt.plot(y_test_scaled, color='black', label='test')
plt.plot(y_pred_scaled, color='green', label='pred')
plt.legend()
plt.show()

model.save(ticker + '_model.keras')
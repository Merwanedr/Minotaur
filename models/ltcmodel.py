import datetime
from binance.client import Client
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# Instanciate Binance client
client = Client('API_KEY', 'SECRET_KEY')

# get data
symbol = 'LTCUSDT'
LTC = client.get_historical_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_30MINUTE, start_str="1 year ago UTC")
LTC = pd.DataFrame(LTC, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])

LTC['Open time'] = pd.to_datetime(LTC['Open time'], unit='ms')

LTC.set_index('Open time', inplace=True)

LTC['Close']=LTC['Close'].astype(float)

data = LTC.iloc[:,3:4].astype(float).values

scaler= MinMaxScaler()
data= scaler.fit_transform(data)

training_set = data[:10000]
test_set = data[10000:]
# Data preprocessing (Dividing datasets to training and testing data)
X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]

X_test = test_set[0:len(test_set)-1]
y_test = test_set[1:len(test_set)]

X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))

# Init the model
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(256))
model.add(Dense(1))
# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=50, batch_size=16, shuffle=False)
# Save the model
model.save('litecoin_model.h5')

# Perform predictions on test data
predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)
real_price = scaler.inverse_transform(y_test)

# Display graph of our prediction
plt.figure(figsize=(10,4))
red_patch = mpatches.Patch(color='red', label='Predicted Price of Litcoin')
blue_patch = mpatches.Patch(color='blue', label='Real Price of Litcoin')
plt.legend(handles=[blue_patch, red_patch])
plt.plot(predicted_price, color='red', label='Predicted Price of Litcoin')
plt.plot(real_price, color='blue', label='Real Price of Litcoin')
plt.title('Predicted vs. Real Price of Litcoin')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

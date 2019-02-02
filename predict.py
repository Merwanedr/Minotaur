import datetime
from binance.client import Client
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import load_model
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys

# Instanciate Binance client
client = Client('API_KEY', 'SECRET_KEY')

if sys.argv[1] == 'bitcoin':
    symbol = 'BTCUSDT'
elif sys.argv[1] == 'ethereum':
    symbol = 'ETHUSDT'
elif sys.argv[1] == 'ripple':
    symbol = 'XRPUSDT'
elif sys.argv[1] == 'litecoin':
    symbol = 'LTCUSDT'
else:
    print(sys.argv[1]+' doesn\'t exist or isn\'t implemented yet.')
    sys.exit()

# get data
symbol = 'ETHUSDT'
CRYPTOCURRENCY = client.get_historical_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_30MINUTE, start_str="1 year ago UTC")
CRYPTOCURRENCY = pd.DataFrame(CRYPTOCURRENCY, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])

CRYPTOCURRENCY['Open time'] = pd.to_datetime(CRYPTOCURRENCY['Open time'], unit='ms')

CRYPTOCURRENCY.set_index('Open time', inplace=True)

CRYPTOCURRENCY['Close']=CRYPTOCURRENCY['Close'].astype(float)

data = CRYPTOCURRENCY.iloc[:,3:4].astype(float).values

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

model = load_model(sys.argv[1]+'_model.h5')
model.compile(loss='mean_squared_error', optimizer='adam')

# Perform predictions on test data
predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)
real_price = scaler.inverse_transform(y_test)

# Display graph of our prediction
plt.figure(figsize=(10,4))
red_patch = mpatches.Patch(color='red', label='Predicted Price of '+sys.argv[1])
blue_patch = mpatches.Patch(color='blue', label='Real Price of '+sys.argv[1])
plt.legend(handles=[blue_patch, red_patch])
plt.plot(predicted_price, color='red', label='Predicted Price of '+sys.argv[1])
plt.plot(real_price, color='blue', label='Real Price of '+sys.argv[1])
plt.title('Predicted vs. Real Price of '+sys.argv[1])
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

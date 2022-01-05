import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pandas_datareader as dr
from datetime import date
import stock_models as model
from tensorflow import config
import os.path  # to save model

# get stock data from 'start' to 'end' dates
start_date = "1950-01-01"
curr_date = date.today()
df = dr.DataReader('BTC-USD', 'yahoo', start=start_date, end=curr_date)

# keep only the closing prices
data = df.filter(['Close'])

# normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = pd.DataFrame(scaler.fit_transform(data))
scaled_data = scaled_data.rename(columns={0: 'Close'})

# use 75% of the data to train the model, 25% to test
train_data_size = int(0.75 * len(data))
train_data = scaled_data[:train_data_size]
test_data = scaled_data[train_data_size:]

# make array from train_data dataframe
train_data_array = np.array(train_data)

# split train data into inputs and targets
trail = 90  # number of days used to make prediction
x_train, y_train = [], []
for i in range(train_data_array.shape[0]-trail):
    x_train.append(train_data_array[i:i+trail])   # input will be closing prices from last 90 days
    y_train.append(train_data_array[i+trail, 0])  # target will be closing price one day after

# convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# adjust test data to include the last 90 days of train_data
test_data_adj = train_data.tail(trail).append(test_data)

# prepare array of test_data to make inputs and targets
test_data_array = np.array(test_data_adj)
x_test, y_test = [], []
for i in range(test_data_array.shape[0]-trail):
    x_test.append(test_data_array[i:i+trail])
    y_test.append(test_data_array[i+trail])

# prepare inputs and targets to be read
x_test, y_test = np.array(x_test), np.array(y_test)

# build model
# model = m.model_dis()

# m = Sequential()
# m.add(LSTM(units=75, return_sequences=False, input_shape=(trail, 1)))
# m.add(Dense(units=75))
# m.add(Dense(units=75))
# m.add(Dense(units=75))
# m.add(Dense(units=75))
# m.add(Dense(units=75))
# m.add(Dense(units=75))
# m.add(Dense(units=75))
# m.add(Dense(units=75))
# m.add(Dense(units=75))
# m.add(Dense(units=75))
# m.add(Dense(units=75))
# m.add(Dense(units=75))
# m.add(Dense(units=75))
# m.add(Dense(units=1))


model = Sequential()
model.add(LSTM(units=250, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=500))
model.add(Dense(1))
# train the model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=3)

# make predictions on test data
predictions = model.predict(x_test)

# unscale the predictions and targets
predictions_unscaled = scaler.inverse_transform(predictions)
y_test_unscaled = scaler.inverse_transform(y_test)

# compute root mean squared error
rms_error = np.sqrt(np.mean(predictions_unscaled - y_test_unscaled)**2)

# prepare data to be plotted
train = data[:train_data_size]
test = data[train_data_size:]  # should have same length as predictions
test.insert(loc=1, column='Predictions', value=predictions_unscaled)

plt.figure(figsize=(14, 7))
plt.title('Closing Prices', fontsize=25)
plt.xlabel('Date', fontsize=15), plt.ylabel('Price (USD $)', fontsize=15)
plt.plot(test[['Close', 'Predictions']])
plt.legend(['Actual', 'Prediction'], loc="upper left")

print(rms_error)
plt.show()


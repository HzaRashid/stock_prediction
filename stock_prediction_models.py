from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as dr
import pandas as pd
import numpy as np
from datetime import date

stock_list = ['TSLA', 'NVDA', 'DIS', 'AAPL', '^GSPC']

# get stock data from 'start' to 'end' dates
start_date = "1900-01-01"
curr_date = date.today()
trail = 90  # number of days used to make prediction

'''
Takes as input a String containing a stock ticker symbol. 
Gathers historical information of the stock from start_date 
to curr_date, only keeping data of the closing prices. Then 
forms two sets, training and test sets, that are both split
into inputs and targets on which a model will be trained. 
Returns a tuple of the inputs and targets as numpy arrays.
'''
def get_data(ticker):
    # get stock data, only keep the closing prices
    df = dr.DataReader(ticker, 'yahoo', start=start_date, end=curr_date)
    data = df.filter(['Close'])

    train_data_size = int(0.75 * len(data))

    # normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = pd.DataFrame(scaler.fit_transform(data))
    scaled_data = scaled_data.rename(columns={0: 'Close'})

    # use 75% of data to train model, 25% to test
    train_data_size = int(0.75 * len(data))
    train_data = scaled_data[:train_data_size]

    # need last 90 days of train data to predict 1st day of test data
    # test_data = scaled_data[train_data_size:]
    # test_data = train_data.tail(trail).append(test_data)

    # convert train & test data to numpy arrays
    train_data_array = np.array(train_data)
    # test_data_array = np.array(test_data)

    # split train data into inputs and targets
    x_train, y_train = [], []
    for i in range(train_data_array.shape[0] - trail):
        x_train.append(train_data_array[i:i + trail])   # input will be closing prices from last 90 days
        y_train.append(train_data_array[i + trail, 0])  # target will be closing price one day after

    # split test data into inputs and targets
    # x_test, y_test = [], []
    # for i in range(test_data_array.shape[0] - trail):
    #     x_test.append(test_data_array[i:i + trail])
    #     y_test.append(test_data_array[i + trail])

    # inputs and targets to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    # x_test, y_test = np.array(x_test), np.array(y_test)

    return x_train, y_train


def model_aapl():
    model = Sequential()
    model.add(LSTM(units=45, input_shape=(x_train.shape[1], 1)))
    model.add(Dense(units=30))
    model.add(Dense(units=10))
    model.add(Dense(units=10))
    model.add(Dense(units=10))
    model.add(Dense(units=10))
    model.add(Dense(units=10))
    model.add(Dense(units=10))
    model.add(Dense(units=10))
    model.add(Dense(units=10))
    model.add(Dense(units=1))
    return model

def model_nvda():
    model = Sequential()
    model.add(LSTM(units=45, input_shape=(x_train.shape[1], 1)))
    model.add(Dense(units=60))
    model.add(Dense(units=30))
    model.add(Dense(units=10))
    model.add(Dense(units=10))
    model.add(Dense(units=1))
    return model


def model_dis():
    model = Sequential()
    model.add(LSTM(units=32, input_shape=(x_train.shape[1], 1)))
    model.add(Dense(units=64))
    model.add(Dense(units=32))
    model.add(Dense(units=32))
    model.add(Dense(units=16))
    model.add(Dense(units=8))
    model.add(Dense(units=1))
    return model

def model_tsla():
    model = Sequential()
    model.add(LSTM(units=54, input_shape=(x_train.shape[1], 1)))
    model.add(Dense(units=256))
    model.add(Dense(units=512))
    model.add(Flatten())
    model.add(Dense(units=1))
    return model


def model_gspc():  # ^GSPC
    model = Sequential()
    model.add(LSTM(units=52, input_shape=(x_train.shape[1], 1)))
    model.add(Dense(units=1))
    return model


optimal_models = {'TSLA': model_tsla(),
                  'NVDA': model_nvda(),
                  'AAPL': model_aapl(),
                  'DIS': model_dis()
                  }

for ticker in optimal_models:
    x_train, y_train = get_data(ticker)
    model = optimal_models.get(ticker)

from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as dr
from datetime import date
import stock_models as m
import pandas as pd
import numpy as np
import os.path

stock_list = ['TSLA', 'NVDA', 'DIS', 'AAPL', '^GSPC']

# get stock data from between these dates
start_date = "1900-01-01"
curr_date = date.today()

trail = 90  # number of days used to make prediction


'''
Takes as input a String containing a stock ticker symbol. 
Gathers historical information of the stock from start_date 
to curr_date, only keeping data of the closing prices. 

Then forms two sets, training and test sets, that are both split
into inputs and targets on which a model will be trained.
 
Returns a tuple of the inputs and targets as numpy arrays.
'''
def get_data(ticker):
    # get stock data, only keep the closing prices
    df = dr.DataReader(ticker, 'yahoo', start=start_date, end=curr_date)
    data = df.filter(['Close'])

    # normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = pd.DataFrame(scaler.fit_transform(data))
    scaled_data = scaled_data.rename(columns={0: 'Close'})

    # use 75% of data to train model, 25% to test
    train_data_size = int(0.75 * len(data))
    train_data = scaled_data[:train_data_size]

    # convert train data to numpy array
    train_data_array = np.array(train_data)

    # split train data into inputs and targets
    inputs, targets = [], []
    for i in range(train_data_array.shape[0] - trail):
        inputs.append(train_data_array[i:i + trail])    # input will be closing prices from last 90 days
        targets.append(train_data_array[i + trail, 0])  # target will be closing price one day after

    # turn inputs and targets to numpy arrays
    inputs, targets = np.array(inputs), np.array(targets)

    return inputs, targets


optimal_models = {'TSLA': m.model_tsla(),
                  'NVDA': m.model_nvda(),
                  'AAPL': m.model_aapl(),
                  'DIS': m.model_dis(),
                  '^GSPC': m.model_gspc()
                  }


if __name__ == "__main__":
    # train all models and save them
    i = 0
    for ticker in optimal_models:

        model = optimal_models.get(ticker)
        model.compile(loss='mean_squared_error', optimizer='adam')

        x_train, y_train = get_data(ticker)
        model.fit(x_train, y_train, epochs=5)
        i += 1
        print(str(i) + '/5 complete')

        # save the model with this filename
        filename = ticker + '_model.h5'

        if not os.path.isfile(filename):
            model.save(filename)

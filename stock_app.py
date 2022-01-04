import datetime
import streamlit as st
import pandas as pd
import pandas.core.frame
import plotly
from plotly import graph_objs as go
import numpy as np
import pandas_datareader as dr
from keras.models import load_model
import datetime as dt
from save_models import start_date, curr_date, stock_list, trail
from sklearn.preprocessing import MinMaxScaler

st.title('Model Performance (Closing Prices)')

ticker_options = stock_list

user_choice_ticker = st.selectbox('Enter ticker', ticker_options)
# st.subheader("Select start and end dates for the model's predictions:")


@st.cache
def get_test_data(ticker):
    # get stock data, only keep the closing prices
    df = dr.DataReader(ticker, 'yahoo', start=start_date, end=curr_date)

    train_portion = int(0.75 * len(df))  # size of data used to train model
    df = df.filter(['Close'])
    data = df[train_portion-trail:]

    # normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data_array = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaler.fit_transform(data))

    test_data = scaled_data.rename(columns={0: 'Close'})
    test_data_array = np.array(test_data)

    # split test data into inputs and targets
    inputs, targets = [], []
    for i in range(test_data_array.shape[0] - trail):
        inputs.append(test_data_array[i:i + trail])
        targets.append(test_data_array[i + trail])

    inputs, targets = np.array(inputs), np.array(targets)

    filename = str(user_choice_ticker).upper() + '_model.h5'
    model = load_model(filename)

    scaled_predictions = model.predict(inputs)
    predictions = scaler.inverse_transform(scaled_predictions)
    data = data[trail:]
    data.insert(loc=1, column='Predictions', value=predictions)
    data.index = data.index.date
    data = data.rename(columns={'Close': 'Actual'})

    return data


data_load_state = st.text('...')
ticker_data = get_test_data(user_choice_ticker)
data_load_state.text('')

# st.write(ticker_data.tail())

# def plot_test_data():
#     graph = go.Figure
#     graph.add_trace(go.Scatter(xaxis=))


if __name__ == "__main__":
    data = get_test_data('TSLA')
    
    # data.index = data.index.date
    # print(data.tail())
    # df = dr.DataReader('TSLA', 'yahoo', start=start_date, end=curr_date)
    # train_data_size = int(0.75 * len(df))
    # data = df.filter(['Close'][train_data_size:])
    # print('all good')
    # data = get_test_data(user_choice_ticker)
    # print(type(start_date))

    # predictions =
    #
    #
    # def plot():
    #     graph = go.Figure()

    # if end_date > today:
    #     end_date = today
    #
    # if start_date > end_date:
    #     temp_date = start_date
    #     start_date = end_date
    #     end_date = temp_date
    #
    # # need the last 'trail' number of days a make a prediction on the first day
    # start_date = start_date - datetime.timedelta(days=trail)



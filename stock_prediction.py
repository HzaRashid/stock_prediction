from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense
from keras.models import Sequential
from plotly import graph_objs as go
import pandas_datareader as dr
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np


st.title('Visualizing Model Performance')
st.subheader("Closing Price Predictions")

ticker_options = ['TSLA', 'NVDA', 'DIS', 'AAPL', 'BTC-USD']

user_choice_ticker = st.selectbox('Enter ticker', ticker_options)

start_date = "1900-01-01"
curr_date = date.today()
trail = 90

# create models
def model_aapl():
    m = Sequential()
    m.add(LSTM(units=45, input_shape=(trail, 1)))
    m.add(Dense(units=30))
    m.add(Dense(units=20))
    m.add(Dense(units=10))
    m.add(Dense(units=10))
    m.add(Dense(units=1))
    return m


def model_nvda():
    m = Sequential()
    m.add(LSTM(units=45, input_shape=(trail, 1)))
    m.add(Dense(units=60))
    m.add(Dense(units=30))
    m.add(Dense(units=10))
    m.add(Dense(units=10))
    m.add(Dense(units=1))
    return m


def model_dis():
    m = Sequential()
    m.add(LSTM(units=48, input_shape=(trail, 1)))
    m.add(Dense(units=1))
    return m


def model_tsla():
    m = Sequential()
    m.add(LSTM(units=256, input_shape=(trail, 1)))
    m.add(Dense(units=1024))
    m.add(Dense(units=1))
    return m


def model_btc():
    m = Sequential()
    m.add(LSTM(units=512, input_shape=(trail, 1)))
    m.add(Dense(1))
    return m


# store tickers and corresponding models
optimal_models = {'TSLA': model_tsla(),
                  'NVDA': model_nvda(),
                  'AAPL': model_aapl(),
                  'DIS': model_dis(),
                  'BTC-USD': model_btc()
                  }


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_test_data(ticker):
    # get stock data, only keep the closing prices
    df = dr.DataReader(ticker, 'yahoo', start=start_date, end=curr_date)
    raw_data = df.filter(['Close'])
    raw_data = raw_data.replace(',', '', regex=True)

    # normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = pd.DataFrame(scaler.fit_transform(raw_data))
    scaled_data = scaled_data.rename(columns={0: 'Close'})

    # use 75% of data to train model, 25% to test
    train_data_size = int(0.75 * len(raw_data))
    train_data = scaled_data[:train_data_size]

    # subtract tail since last 90 days is needed to predict 1st day of test data
    test_data = scaled_data[train_data_size-trail:]
    test_data = test_data.rename(columns={0: 'Close'})

    # convert train and test data to numpy arrays
    train_data_array = np.array(train_data)
    test_data_array = np.array(test_data)

    # split train data into inputs and targets
    x_train, y_train = [], []
    for i in range(train_data_array.shape[0] - trail):
        x_train.append(train_data_array[i:i + trail])    # input will be closing prices from last 90 days
        y_train.append(train_data_array[i + trail, 0])   # target will be closing price one day after

    x_train, y_train = np.array(x_train), np.array(y_train)

    # split test data into inputs and targets
    x_test, y_test = [], []
    for i in range(test_data_array.shape[0] - trail):
        x_test.append(test_data_array[i:i + trail])
        y_test.append(test_data_array[i + trail])

    # convert to numpy arrays to be read for training
    x_test, y_test = np.array(x_test), np.array(y_test)

    # get corresponding model for the stock and train it
    model = optimal_models.get(ticker)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=3)

    # unscale the predictions to they can be inserted into to the test data
    scaled_predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(scaled_predictions)

    data = raw_data[train_data_size:]
    data.insert(loc=1, column='Prediction', value=predictions)
    data.index = data.index.date
    data = data.rename(columns={'Close': 'Actual'})

    # compute error of all predictions
    errors = []
    for i in range(data.shape[0]):
        errors.append(abs(data['Prediction'][i]-data['Actual'][i]))
    data.insert(loc=2, column='Error', value=errors)

    return data


data_load_state = st.text('Training model...')
ticker_data = get_test_data(user_choice_ticker)
data_load_state.text('')

ticker_data = ticker_data.replace(',', '', regex=True)

# plot Actual and Predicted closing prices
def plot_data():
    fig = go.Figure([
        go.Scatter(
            x=ticker_data.index,
            y=ticker_data['Actual'],
            name='Actual Closing Price',
            mode='lines',
            line=dict(color='#088F8F')
        ),
        go.Scatter(
            x=ticker_data.index,
            y=ticker_data['Prediction'],
            name='Predicted Closing Price',
            mode='lines',
            line=dict(color='#F08080')
        )
    ])
    fig.layout.update(
        title_text='Actual and Predicted Prices ($USD)',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        xaxis_rangeslider_visible=True
    )
    return fig


st.plotly_chart(plot_data())

# plot error
def plot_error():
    fig = go.Figure([
        go.Scatter(
            x=ticker_data.index,
            y=ticker_data['Error'],
            mode='lines',
            line=dict(color='#F08080')
        )
    ])

    fig.layout.update(
        title_text='Prediction Error (Absolute Difference Between Actual and Predicted Prices, $USD)',
        xaxis_title='Date',
        yaxis_title='Difference ($)',
        xaxis_rangeslider_visible=True
    )
    return fig


st.plotly_chart(plot_error())

# show raw data
st.subheader('Raw Data ($USD)')
st.write(ticker_data)

if __name__ == "__main__":
    data = get_test_data('TSLA')
    print(data)
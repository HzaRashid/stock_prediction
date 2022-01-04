from save_models import start_date, curr_date, stock_list, trail
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from plotly import graph_objs as go
import pandas_datareader as dr
import streamlit as st
import pandas as pd
import numpy as np


st.title("Model Performance (Closing Prices)")

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


def plot_data():
    plot = go.Figure()
    plot.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Actual'], name='Actual Closing Price'))
    plot.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Predictions'], name='Predicted Closing Price'))
    plot.layout.update(title_text='Actual and Predicted Closing Prices', xaxis_rangeslider_visible=True)
    return plot


st.plotly_chart(plot_data())



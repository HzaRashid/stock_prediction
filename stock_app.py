from plotly import graph_objs as go
import streamlit as st
import pandas as pd
import os

st.title('Visualizing Model Performance')
st.subheader("Closing Price Predictions")

ticker_options = ['TSLA', 'NVDA', 'DIS', 'AAPL', 'BTC-USD']

user_choice_ticker = st.selectbox('Enter ticker', ticker_options)

# create dictionary mapping tickers to corresponding model's test data
@st.cache(show_spinner=False)
def load_data():
    model_data = {}
    for ticker in ticker_options:

        # get filename and path of test data
        filename = ticker + '_model_data.csv'
        fpath = os.path.abspath(filename)

        # create dataframe

        ticker_data = pd.read_csv(fpath)
        ticker_data = ticker_data.rename(columns={'Unnamed: 0': 'Date'})
        ticker_data.index = ticker_data['Date']
        ticker_data = ticker_data.drop(columns={'Date'})
        # ticker_data.reset_index(inplace=True)

        # map the ticker to corresponding data
        model_data[ticker] = ticker_data

    return model_data


data = load_data().get(user_choice_ticker)

def plot_data():
    fig = go.Figure([
        go.Scatter(
            x=data.index,
            y=data['Actual'],
            name='Actual Closing Price',
            mode='lines',
            line=dict(color='#088F8F')
        ),
        go.Scatter(
            x=data.index,
            y=data['Prediction'],
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
            x=data.index,
            y=data['Error'],
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
st.write(data)

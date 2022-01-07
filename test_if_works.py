from plotly import graph_objs as go
from datetime import date
import streamlit as st
import pandas as pd


st.title('Visualizing Model Performance')
st.subheader("Closing Price Predictions")

ticker_options = ['TSLA', 'NVDA', 'DIS', 'AAPL', 'BTC-USD']

user_choice_ticker = st.selectbox('Enter ticker', ticker_options)

start_date = "1950-01-01"
curr_date = date.today()
trail = 90

@st.cache(allow_output_mutation=True, show_spinner=False)
def get_test_data(ticker):
    return pd.read_csv(ticker + '_model_data.csv')


data_load_state = st.text('Training model...')
data = get_test_data(user_choice_ticker)
data_load_state.text('')

# plot Actual and Predicted closing prices
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

if __name__ == '__main__':
    data = get_test_data('TSLA')
    print(data)

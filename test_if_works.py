from plotly import graph_objs as go
from datetime import date
import streamlit as st
import pandas as pd
from pathlib import Path

# tsla = Path(__file__).parents[1]/'stock_models/TSLA_model_data.csv'

tsla_file = Path(__file__).parents[1] / 'stock_models/TSLA_model_data.csv'
# nvda_file = Path(__file__).parents[1] / 'stock_models/NVDA_model_data.csv'
# dis_file = Path(__file__).parents[1] / 'stock_models/DIS_model_data.csv'
# aapl_file = Path(__file__).parents[1] / 'stock_models/AAPL_model_data.csv'
# btc_file = Path(__file__).parents[1] / 'stock_models/BTC-USD_model_data.csv'

# file:///Users/hamzarashid/tensorflow-test/TSLA_model_data.csv
# file:///Users/hamzarashid/tensorflow-test/NVDA_model_data.csv
# file:///Users/hamzarashid/tensorflow-test/DIS_model_data.csv
# file:///Users/hamzarashid/tensorflow-test/AAPL_model_data.csv
# file:///Users/hamzarashid/tensorflow-test/BTC-USD_model_data.csv
file_options = {'TSLA': tsla_file,
                }

st.title('Visualizing Model Performance')
st.subheader("Closing Price Predictions")

ticker_options = ['TSLA', 'NVDA', 'DIS', 'AAPL', 'BTC-USD']

user_choice_ticker = st.selectbox('Enter ticker', ['TSLA'])

start_date = "1950-01-01"
curr_date = date.today()
trail = 90

# file_options = ('TSLA_model_data.csv',
#                 'NVDA_model_data.csv',
#                 'DIS_model_data.csv',
#                 'AAPL_model_data.csv',
#                 'BTC-USD_model_data.csv')

# uploaded_files = st.file_uploader('Select data', file_options)
# for file in uploaded_files:
#     bytes_data = file.read()
#     st.write('filename: ', file.name[:-15])
#     st.write(bytes_data)

@st.cache
def get_data(ticker):
    file = file_options.get(ticker)
    data = pd.read_csv(file)
    data = data.rename(columns={'Unnamed: 0': 'Date'})
    data.index = data['Date']
    data = data.drop(columns={'Date'})
    data.reset_index(inplace=True)
    return data

df = get_data(user_choice_ticker)
st.subheader(user_choice_ticker)
st.dataframe(df)
# @st.cache(allow_output_mutation=True, show_spinner=False)
# def get_test_data(ticker):
#     # file = st.file_uploader(ticker + '_model_data.csv')
#     df = pd.read_csv(ticker + '_model_data.csv')
#
#     # df = df.rename(columns={'Close': 'Actual'})
#     return df


# data_load_state = st.text('Training model...')
# data = get_test_data(user_choice_ticker).rename(columns={'Close': 'Actual'})
#
# data_load_state.text('')
#
# # plot Actual and Predicted closing prices
# def plot_data():
#     fig = go.Figure([
#         go.Scatter(
#             x=data.index,
#             y=data['Actual'],
#             name='Actual Closing Price',
#             mode='lines',
#             line=dict(color='#088F8F')
#         ),
#         go.Scatter(
#             x=data.index,
#             y=data['Prediction'],
#             name='Predicted Closing Price',
#             mode='lines',
#             line=dict(color='#F08080')
#         )
#     ])
#     fig.layout.update(
#         title_text='Actual and Predicted Prices ($USD)',
#         xaxis_title='Date',
#         yaxis_title='Price ($)',
#         xaxis_rangeslider_visible=True
#     )
#     return fig
#
#
# st.plotly_chart(plot_data())
#
#
# # plot error
# def plot_error():
#     fig = go.Figure([
#         go.Scatter(
#             x=data.index,
#             y=data['Error'],
#             mode='lines',
#             line=dict(color='#F08080')
#         )
#     ])
#
#     fig.layout.update(
#         title_text='Prediction Error (Absolute Difference Between Actual and Predicted Prices, $USD)',
#         xaxis_title='Date',
#         yaxis_title='Difference ($)',
#         xaxis_rangeslider_visible=True
#     )
#     return fig
#
#
# st.plotly_chart(plot_error())
#
# # show raw data
# st.subheader('Raw Data ($USD)')
# st.write(data)

if __name__ == '__main__':
    # tsla = Path(__file__).parents[1] / 'stock_models/TSLA_model_data.csv'
    # print(tsla)
    # tsla_file = 'file://' + Path(__file__).parents[1] / 'stock_models/TSLA_model_data.csv'
    data = get_data('TSLA')
    print(data)
from plotly import graph_objs as go
from datetime import date
import streamlit as st
import pandas as pd
from pathlib import Path

st.title('Visualizing Model Performance')
st.subheader("Closing Price Predictions")

ticker_options = ['TSLA', 'NVDA', 'DIS', 'AAPL', 'BTC-USD']

user_choice_ticker = st.selectbox('Enter ticker', ['TSLA'])

start_date = "1950-01-01"
curr_date = date.today()
trail = 90

tsla_file = Path(__file__).parents[1] / '/app/tensorflow-test/TSLA_model_data.csv'
df = pd.read_csv(filepath_or_buffer=tsla_file)
st.subheader(user_choice_ticker)
st.dataframe(df)

if __name__ == '__main__':
    # tsla = Path(__file__).parents[1] / 'stock_models/TSLA_model_data.csv'
    # print(tsla)
    # tsla_file = 'file://' + Path(__file__).parents[1] / 'stock_models/TSLA_model_data.csv'
    # data = get_data('TSLA')
    print(df)






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
    # tsla = Path(__file__).parents[1]/'stock_models/TSLA_model_data.csv'

    # tsla_file = Path(__file__).parents[1] / 'stock_models/TSLA_model_data.csv'
    # tsla_file = '/Users/hamzarashid/tensorflow-test/TSLA_model_data.csv'
    # print(tsla_file)
    # nvda_file = Path(__file__).parents[1] / 'stock_models/NVDA_model_data.csv'
    # dis_file = Path(__file__).parents[1] / 'stock_models/DIS_model_data.csv'
    # aapl_file = Path(__file__).parents[1] / 'stock_models/AAPL_model_data.csv'
    # btc_file = Path(__file__).parents[1] / 'stock_models/BTC-USD_model_data.csv'

    # file:///Users/hamzarashid/tensorflow-test/TSLA_model_data.csv
    # file:///Users/hamzarashid/tensorflow-test/NVDA_model_data.csv
    # file:///Users/hamzarashid/tensorflow-test/DIS_model_data.csv
    # file:///Users/hamzarashid/tensorflow-test/AAPL_model_data.csv
    # file:///Users/hamzarashid/tensorflow-test/BTC-USD_model_data.csv

# @st.cache
# def get_data(ticker):
#     file = file_options.get(ticker)
#     data = pd.read_csv(file)
#     data = data.rename(columns={'Unnamed: 0': 'Date'})
#     data.index = data['Date']
#     data = data.drop(columns={'Date'})
#     data.reset_index(inplace=True)
#     return data


# tsla_csv = Path(__file__).parents[1] / 'stock_models/TSLA_model_data.csv'
# print(tsla_csv)

# file_options = {'TSLA': tsla_csv,
#                 }

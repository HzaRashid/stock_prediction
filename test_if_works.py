from plotly import graph_objs as go
from datetime import date
import streamlit as st
import pandas as pd
from pathlib import Path

st.title('Visualizing Model Performance')
st.subheader("Closing Price Predictions")

ticker_options = ['TSLA', 'NVDA', 'DIS', 'AAPL', 'BTC-USD']

user_choice_ticker = st.selectbox('Enter ticker', ticker_options)

@st.cache(show_spinner=False)
def load_data():
    model_data = {}
    for ticker in ticker_options:

        # get filename and path of test data
        filename = ticker + '_model_data.csv'
        fpath = Path(__file__).parent/filename

        # create dataframe
        data = pd.read_csv(fpath)
        data = data.rename(columns={'Unnamed: 0': 'Date'})
        data.index = data['Date']
        data = data.drop(columns={'Date'})
        data.reset_index(inplace=True)

        # map the ticker to corresponding data
        model_data[ticker] = data

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

if __name__ == "__main__":
    print(load_data().get('BTC-USD'))






# from plotly import graph_objs as go
# from datetime import date
# import streamlit as st
# import pandas as pd
# from pathlib import Path
#
# st.title('Visualizing Model Performance')
# st.subheader("Closing Price Predictions")
#
# ticker_options = ['TSLA', 'NVDA', 'DIS', 'AAPL', 'BTC-USD']
#
# user_choice_ticker = st.selectbox('Enter ticker', ticker_options)
#
# start_date = "1950-01-01"
# curr_date = date.today()
# trail = 90
#
# @st.cache(show_spinner=False)
# def load_data():
#     model_data = {}
#     for ticker in ticker_options:
#
#         # get filename and path of test data
#         filename = ticker + '_model_data.csv'
#         fpath = Path(__file__).parent/filename
#
#         # create dataframe
#         data = pd.read_csv(fpath)
#         data = data.rename(columns={'Unnamed: 0': 'Date'})
#         data.index = data['Date']
#         data = data.drop(columns={'Date'})
#         data.reset_index(inplace=True)
#
#         # map the ticker to corresponding data
#         model_data[ticker] = data
#
#     return model_data
#
#
# data = load_data().get(user_choice_ticker)
# st.write(data)
# print(data)


# if __name__ == '__main__':
    # tsla = Path(__file__).parents[1] / 'stock_models/TSLA_model_data.csv'
    # print(tsla)
    # tsla_file = 'file://' + Path(__file__).parents[1] / 'stock_models/TSLA_model_data.csv'
    # data = get_data('TSLA')
    # print(df)






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

# file_options = {'TSLA': tsla_csv, }
# from plotly import graph_objs as go
# from datetime import date
# import streamlit as st
# import pandas as pd
# import pandas_datareader as dr
# from pathlib import Path
#
# st.title('Visualizing Model Performance')
# st.subheader("Closing Price Predictions")
#
# ticker_options = ['TSLA', 'NVDA', 'DIS', 'AAPL', 'BTC-USD']
#
# user_choice_ticker = st.selectbox('Enter ticker', ['TSLA'])
#
# start_date = "1950-01-01"
# curr_date = date.today()
# trail = 90
#
# tsla_file = Path(__file__).parents[1] / 'TSLA_model_data.csv'
# fuck = pd.read_csv('TSLA_model_data.csv')
# st.write(fuck)
# print(tsla_file)
# df = pd.read_csv(filepath_or_buffer=tsla_file)
# st.subheader(user_choice_ticker)
# st.dataframe(df)

# if __name__ == '__main__':
    # tsla = Path(__file__).parents[1] / 'stock_models/TSLA_model_data.csv'
    # print(tsla)
    # tsla_file = 'file://' + Path(__file__).parents[1] / 'stock_models/TSLA_model_data.csv'
    # data = get_data('TSLA')
    # print(df)






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

# file_options = {'TSLA': tsla_csv, }
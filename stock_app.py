import streamlit as st
import pandas as pd
import numpy as np
from pandas_datareader import data as dr
from keras.models import load_model

model = load_model(filepath='models/stock_model.h5')

st.title('Stock Price Predictor')

ticker_options = pd.read_csv('stock_tickers.csv')
user_choice_ticker = st.selectbox('Enter ticker', ticker_options)
st.write("Can't find the ticker? \
         Search the company's name [here](https://ca.finance.yahoo.com/) \
         and the ticker should appear!")


@st.cache
def get_data(ticker):
    df = dr.get_data_yahoo(user_choice_ticker)
    return df


data = get_data(user_choice_ticker)
st.subheader('Past 5 Days Raw Data')
st.write(data.tail(5))

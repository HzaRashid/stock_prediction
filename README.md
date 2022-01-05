# stock_prediction

Gather historical data of a stock, organize it into training and test sets, and build an [RNN](https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9) to predict its closing prices.

I chose 4 stocks – Tesla, Nvidia, Apple, Disney – and Bitcoin, and made a model for each one. The models were trained on 75% of the corresponding stock's closing price data, and tested on the remainder. The entire project can be found in test_deploy.py. 

### The visualization can be found [here](https://share.streamlit.io/hzarashid/stock_prediction/main/test_deploy.py) 

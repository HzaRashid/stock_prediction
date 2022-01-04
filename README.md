# stock_prediction

Gather historical data of a stock, organize it into training and test sets, and build an [RNN](https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9) to predict its closing prices.

I chose 4 stocks – Tesla, Nvidia, Apple, Disney – and the S&P 500 index, and made a model for each one. The models are located in [stock_models.py](https://github.com/HzaRashid/stock_prediction/blob/main/stock_models.py). The models were trained in [save_models.py](https://github.com/HzaRashid/stock_prediction/blob/main/save_models.py) (using 75% of the corresponding stock's closing price data), and saved there as well. Lastly, the models were tested in [stock_app.py](https://github.com/HzaRashid/stock_prediction/blob/main/stock_app.py) (using the remaining 25% of the data).
[test](https://share.streamlit.io/hzarashid/stock_prediction/main/test_deploy.py)

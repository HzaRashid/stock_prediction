# stock_prediction

#### Gather historical data of a stock, organize it into training and test sets, and build an [RNN](https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9) to predict its closing prices.

I chose 4 stocks – Tesla, Nvidia, Apple, Disney – and Bitcoin, and made a model for each one. The models were trained on 75% of the corresponding stock's closing price data, and tested on the remainder. [test_models.py](https://github.com/HzaRashid/stock_prediction/blob/main/test_models.py) was where I tried out different models for each stock and to see which one's were most accurate. 

The entire project can be found in [stock_prediction.py](https://github.com/HzaRashid/stock_prediction/blob/main/stock_prediction.py). 


#### The visualization can be found [here](https://share.streamlit.io/hzarashid/stock_prediction/main/stock_prediction.py) 

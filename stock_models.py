from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten

trail = 90

def model_aapl():
    m = Sequential()
    m.add(LSTM(units=45, input_shape=(trail, 1)))
    m.add(Dense(units=30))
    m.add(Dense(units=10))
    m.add(Dense(units=10))
    m.add(Dense(units=10))
    m.add(Dense(units=10))
    m.add(Dense(units=10))
    m.add(Dense(units=10))
    m.add(Dense(units=10))
    m.add(Dense(units=10))
    m.add(Dense(units=1))
    return m


def model_nvda():
    m = Sequential()
    m.add(LSTM(units=45, input_shape=(trail, 1)))
    m.add(Dense(units=60))
    m.add(Dense(units=30))
    m.add(Dense(units=10))
    m.add(Dense(units=10))
    m.add(Dense(units=1))
    return m


def model_dis():
    m = Sequential()
    m.add(LSTM(units=32, input_shape=(trail, 1)))
    m.add(Dense(units=64))
    m.add(Dense(units=32))
    m.add(Dense(units=32))
    m.add(Dense(units=16))
    m.add(Dense(units=8))
    m.add(Dense(units=1))
    return m


def model_tsla():
    m = Sequential()
    m.add(LSTM(units=54, input_shape=(trail, 1)))
    m.add(Dense(units=256))
    m.add(Dense(units=512))
    m.add(Flatten())
    m.add(Dense(units=1))
    return m


def model_gspc():  # ^GSPC
    m = Sequential()
    m.add(LSTM(units=52, input_shape=(trail, 1)))
    m.add(Dense(units=1))
    return m
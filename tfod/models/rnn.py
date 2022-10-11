""" 
https://www.renom.jp/id/notebooks/tutorial/time_series/lstm-anomalydetection/notebook.html
"""


import tensorflow as tf
from tensorflow.keras.layers import LSTM, Activation, Dense


class RNN(object):
    def __init__(self, predict_sequence_length=3) -> None:
        self.rnn1 = LSTM(35, return_sequences=True)
        self.relu = Activation("relu")
        self.rnn2 = LSTM(35)
        self.relu = Activation("relu")
        self.dense = Dense(predict_sequence_length)

    def __call__(self, x):
        # x = self.rnn1(x)
        # x = self.relu(x)
        x = self.rnn2(x)
        # x = self.relu(x)
        x = self.dense(x)
        return x

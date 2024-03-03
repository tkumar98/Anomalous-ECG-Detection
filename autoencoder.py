'''
A simple autoencoder architecture using Neural Network
'''

import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.losses import mae
from keras.optimizers import Adam

class Autoencoder(Model):

    def __init__(self, input_size=140):
        super(Autoencoder, self).__init__()

        self.encoder = Sequential([Input(shape=(input_size,)),
                               Dense(units=64, activation='relu'),
                               Dense(units=32, activation='relu'),
                               Dense(units=16, activation='relu'),
                               ])

        self.decoder = Sequential([Dense(units=32, activation='relu'),
                                   Dense(units=64, activation='relu'),
                                   Dense(units=140, activation='linear')])

    def call(self, inputs, training=None):
        encoded = self.encoder(inputs, training=training)
        decoded = self.decoder(encoded, training=training)
        return decoded

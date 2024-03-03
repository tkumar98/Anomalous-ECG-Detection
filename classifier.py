'''
This module contains the classifier that uses a trained autoencoder model to
classify ECG as Normal (1) or Abnormal (0).
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.losses import mae

class Classifier():
    def __init__(self, model, X):
        '''
        :param model: Model to be used for classifying the ECG
        :param X: Data on which model is trained
        '''
        self.model = model
        self.reconstructed = self.model(X)
        self.train_loss = mae(self.reconstructed, X)
        self.t = np.mean(self.train_loss) + np.std(self.train_loss)

    def classify(self, X):
        '''

        :param X: Data which is to be classified
        :return: Array of 0 and 1
        '''
        rec = self.model(X)
        loss = mae(rec, X)

        return tf.cast(tf.math.less(loss, self.t), dtype=tf.int32)



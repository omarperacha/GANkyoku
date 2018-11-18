'''
Created on 18 Nov 2018

@author: omarperacha
'''
from utils import getData
import numpy as np
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import matplotlib.pyplot as plt

X_train = getData()

print(np.shape(X_train))

X_train = X_train.reshape(10, 576, 1)

X_train = X_train.astype('float32')

# Scaling the range of the datapoints to [-1, 1]
# Because we are using tanh as the activation function in the last layer of the generator
# and tanh restricts the weights in the range [-1, 1]
X_train = (X_train - 22) / 22

print(np.unique(X_train))

generator = Sequential([
        Dense(128*7*7, input_dim=100, activation=LeakyReLU(0.2)),
        BatchNormalization(),
        Reshape((49,128)),
        UpSampling1D(),
        Conv1D(64, 5, padding='same', activation=LeakyReLU(0.2)),
        BatchNormalization(),
        UpSampling1D(),
        Conv1D(1, 5, padding='same', activation='tanh')
    ])

generator.summary()


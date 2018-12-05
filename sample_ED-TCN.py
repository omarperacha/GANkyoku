import numpy as np

from keras.models import Model
from keras.layers import Input, TimeDistributed
from keras.layers.core import *
from keras.layers.merge import _Merge
from keras.layers.convolutional import *
import random


import tensorflow as tf
from keras import backend as K

from keras.activations import relu
from functools import partial
from utils import synthData, fromCategoricalNoScaling, getData, getSingleSample
import datetime

clipped_relu = partial(relu, max_value=5)


samples = getData()
NUM_CLASSES = 45
BATCH_SIZE = 350
NUM_EPOCHS = 3000
temp = 4


def max_filter(x):
    # Max over the best filter score (like ICRA paper)
    max_values = K.max(x, 2, keepdims=True)
    max_flag = tf.greater_equal(x, max_values)
    out = x * tf.cast(max_flag, tf.float32)
    return out


def channel_normalization(x):
    # Normalize by the highest activation
    max_values = K.max(K.abs(x), 2, keepdims=True) + 1e-5
    out = x / max_values
    return out


def WaveNet_activation(x):
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)
    return _Merge(mode='mul')([tanh_out, sigm_out])


def ED_TCN(n_nodes, conv_len, n_classes, n_feat, max_len,
           loss='categorical_crossentropy', causal=False,
           optimizer="adam", activation='norm_relu',
           return_param_str=False):
    n_layers = len(n_nodes)

    inputs = Input(shape=(max_len, n_feat))
    model = inputs

    # ---- Encoder ----
    for i in range(n_layers):
        # Pad beginning of sequence to prevent usage of future data
        if causal: model = ZeroPadding1D((conv_len // 2, 0))(model)
        model = Conv1D(n_nodes[i], conv_len, padding='same')(model)
        if causal: model = Cropping1D((0, conv_len // 2))(model)

        model = SpatialDropout1D(0.3)(model)

        if activation == 'norm_relu':
            model = Activation('relu')(model)
            model = Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
        elif activation == 'wavenet':
            model = WaveNet_activation(model)
        else:
            model = Activation(activation)(model)

        model = MaxPooling1D(2)(model)

    model = (Lambda(lambda model: model / temp))(model)

    # ---- Decoder ----
    for i in range(n_layers):
        model = UpSampling1D(2)(model)
        if causal: model = ZeroPadding1D((conv_len // 2, 0))(model)
        model = Conv1D(n_nodes[-i - 1], conv_len, padding='same')(model)
        if causal: model = Cropping1D((0, conv_len // 2))(model)

        model = SpatialDropout1D(0.3)(model)

        if activation == 'norm_relu':
            model = Activation('relu')(model)
            model = Lambda(channel_normalization, name="decoder_norm_{}".format(i))(model)
        elif activation == 'wavenet':
            model = WaveNet_activation(model)
        else:
            model = Activation(activation)(model)

    # Output FC layer
    model = TimeDistributed(Dense(n_classes, activation="softmax"))(model)

    model = Model(input=inputs, output=model)
    model.summary()
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    if return_param_str:
        param_str = "ED-TCN_C{}_L{}".format(conv_len, n_layers)
        if causal:
            param_str += "_causal"

        return model, param_str
    else:
        return model


def getX():
    i = random.randint(0, 9)
    if i % 10 == 0:
        x_train = getSingleSample(samples)
    else:
        x_train = synthData((i % 10)/10, samples)

        x_train = (x_train * 22) + 22

        x_train = np.reshape(x_train, (1, 576, 1))

    return x_train


n_nodes = [64, 96, 128]
conv = 20
NUM_CLASSES = 45
n_feat = 1
max_len = 576

model = ED_TCN(n_nodes, conv, NUM_CLASSES, n_feat, max_len,
                                        activation='norm_relu')

filepath = "weights_ED-TCN/Model2-1468-0.5243.hdf5"
model.load_weights(filepath)

x_train = getX()
# make prediction
probas = model.predict(x_train)
probas = np.reshape(probas, (576, NUM_CLASSES))
pattern = np.zeros((576, 1))

# save sample
for i in range(576):
    prediction = probas[i, :]
    prediction = np.argmax(prediction)
    print(i, prediction)
    pattern[i] = prediction
print("Done.")
pattern = np.reshape(pattern,(576))
pattern = fromCategoricalNoScaling(pattern)
print(pattern)
np.savetxt("samples_ED-TCN/%s.txt" % datetime.datetime.now(), pattern, fmt='%s')



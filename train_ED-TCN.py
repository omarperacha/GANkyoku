import numpy as np

from keras.models import Model
from keras.layers import Input, TimeDistributed
from keras.layers.core import *
from keras.layers.merge import _Merge
from keras.layers.convolutional import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard


import tensorflow as tf
from keras import backend as K

from keras.activations import relu
from functools import partial
from utils import synthData, oneHotEncodeSequence, getData, getSingleSample

clipped_relu = partial(relu, max_value=5)


samples = getData()
NUM_CLASSES = 45
BATCH_SIZE = 350
NUM_EPOCHS = 3000


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


def getXY():
    for j in range(NUM_EPOCHS):
        for i in range(BATCH_SIZE):
            if i % 10 == 0:
                x_train = getSingleSample(samples)
            else:
                x_train = synthData((i % 10)/10, samples)

            y_train = (x_train * 22) + 22

            x_train = np.reshape(y_train, (1, 576, 1))
            y_train = np.reshape(oneHotEncodeSequence(y_train), (1, 576, 45))

            yield x_train, y_train


n_nodes = [64, 96, 128]
conv = 20
NUM_CLASSES = 45
n_feat = 1
max_len = 576

model = ED_TCN(n_nodes, conv, NUM_CLASSES, n_feat, max_len,
                                        activation='norm_relu')

filepath = "weights_ED-TCN/Model2-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=0.0005)
log = TensorBoard(log_dir='./logs_ED-TCN')
callbacks_list = [checkpoint, log, reduce_lr]

gen = getXY()
# fit the model
model.fit_generator(gen, epochs=NUM_EPOCHS, steps_per_epoch=BATCH_SIZE, max_queue_size=1, callbacks=callbacks_list, verbose=2)


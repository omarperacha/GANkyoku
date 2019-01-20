from utils import fromCategorical, pruneNonCandidates

from keras.layers import Input, Dense, Concatenate, Reshape, LSTM
from keras.layers import Dropout
from keras.models import Model


import numpy as np

LOAD_WEIGHTS_PATH = "weights/epoch_17701.h5"
SHOULD_LOAD_WEIGHTS = True
NUM_CONDS = 4

NUM_SAMPLES = 5

latent_dim = 128

def build_generator():

    noise = Input(shape=(latent_dim,))
    condition_tensor = Input(shape=(NUM_CONDS,))
    merged = Concatenate(axis=1)([noise, condition_tensor])

    model = Dense(576, activation="relu", input_dim=(latent_dim + NUM_CONDS))(merged)
    model = Reshape((576, 1))(model)
    model = LSTM(1024, return_sequences=True)(model)
    model = Dropout(0.2)(model)
    model = LSTM(1024, return_sequences=False)(model)
    model = Dropout(0.2)(model)
    model = Dense(576, activation='tanh')(model)
    model = Reshape((576, 1))(model)

    model = Model(inputs=[noise, condition_tensor], outputs=model)

    if SHOULD_LOAD_WEIGHTS:
        model.load_weights(LOAD_WEIGHTS_PATH)

    model.summary()

    return model


generator = build_generator()

def save_samples():
    for i in range(NUM_SAMPLES):
        noise = np.random.normal(0, 1, (1, latent_dim))
        true_class = np.zeros((1, NUM_CONDS))
        true_class[0, 0] = 1
        gen_mus = generator.predict([noise, true_class])
        gen_mus = np.reshape(gen_mus, 576)
        gen_mus = fromCategorical(gen_mus)
        np.savetxt("samples/sample_%i.txt" % (i), gen_mus, fmt='%s')
    #pruneNonCandidates()

save_samples()
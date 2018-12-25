from utils import fromCategorical, pruneNonCandidates

from keras.layers import Input, Dense, Concatenate, Reshape
from keras.layers import BatchNormalization
from keras.models import Model


import numpy as np

LOAD_WEIGHTS_PATH = "weights_TWGAN/epoch_9601.h5"
SHOULD_LOAD_WEIGHTS = True
NUM_CONDS = 4

NUM_SAMPLES = 1

latent_dim = 128

def build_generator():

    noise = Input(shape=(latent_dim,))
    condition_tensor = Input(shape=(NUM_CONDS,))
    merged = Concatenate(axis=1)([noise, condition_tensor])

    model = Dense(256, activation="relu", input_dim=(latent_dim + NUM_CONDS))(merged)
    model = BatchNormalization()(model)
    model = Dense(512)(model)
    model = BatchNormalization()(model)
    model = Dense(1024)(model)
    model = BatchNormalization()(model)
    out = Dense(576, activation='tanh')(model)
    out = Reshape((576, 1))(out)

    model = Model(inputs=[noise, condition_tensor], outputs=out)

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
        np.savetxt("samples_TWGAN/sample_%i.txt" % (i), gen_mus, fmt='%s')
    #pruneNonCandidates()

save_samples()
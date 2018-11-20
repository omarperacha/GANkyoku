'''
Created on 18 Nov 2018

@author: omarperacha
'''

from __future__ import print_function, division
from utils import getData, fromCategorical

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np


class DCGAN():
    def __init__(self):
        # Input shape
        self.inp_rows = 1
        self.inp_cols = 576
        self.channels = 1
        self.inp_shape = (self.inp_rows, self.inp_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        mus = self.generator(z)

        # For the combined model we will train both
        self.discriminator.trainable = True

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(mus)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(576, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((576, 1)))
        model.add(Conv1D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv1D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv1D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        mus = model(noise)

        return Model(noise, mus)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv1D(32, kernel_size=3, padding="same", strides=2, input_shape=(576, 1)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        mus = Input(shape=(576, 1))
        validity = model(mus)

        return Model(mus, validity)

    def train(self, epochs, batch_size=1, save_interval=50):

        X_train = getData()

        print(np.shape(X_train))

        X_train = X_train.reshape(10, 576, 1)

        X_train = X_train.astype('float32')

        # Scaling the range of the datapoints to [-1, 1]
        # Because we are using tanh as the activation function in the last layer of the generator
        # and tanh restricts the weights in the range [-1, 1]
        X_train = (X_train - 22) / 22

        print(np.unique(X_train))

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            idxs = np.random.randint(0, X_train.shape[0], batch_size)
            mus = X_train[idxs]

            # Sample noise and generate a batch of new pieces
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_mus = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(mus, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_mus, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake music as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # If at save interval => save generated music samples
            if epoch % save_interval == 0:
                # Plot the progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
                self.save_samples(epoch)
                self.generator.save_weights("weights/epoch_%d.h5" % epoch)

    def save_samples(self, epoch):
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        gen_mus = self.generator.predict(noise)
        gen_mus = np.reshape(gen_mus, (576))
        gen_mus = fromCategorical(gen_mus)
        np.savetxt("samples/epoch_%d.txt" % epoch, gen_mus, fmt='%s')



if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=200000, batch_size=20, save_interval=500)
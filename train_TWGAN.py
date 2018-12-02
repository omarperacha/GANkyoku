'''
Created on 18 Nov 2018

@author: omarperacha
'''

from __future__ import print_function, division
from utils import getData, fromCategorical, pruneNonCandidates

from keras.layers import Input, Dense, Reshape
from keras.layers.merge import _Merge
from keras.layers import BatchNormalization, Activation
from keras.callbacks import TensorBoard
from keras.layers.convolutional import Conv1D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial
import tensorflow as tf

import keras.backend as K
from tcn import compiled_tcn

import numpy as np

BATCH_SIZE = 20
LOAD_WEIGHTS_PATH = "weights_TWGAN/epoch_0.h5"
SHOULD_LOAD_WEIGHTS = False
SAMPLE_INTERVAL = 25


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated samples"""

    def _merge_function(self, inputs):
        alpha = K.random_uniform((BATCH_SIZE, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class WGAN():
    def __init__(self):
        # Input shape
        self.inp_rows = 1
        self.inp_cols = 576
        self.channels = 1
        self.inp_shape = (self.inp_rows, self.inp_cols, self.channels)
        self.latent_dim = 100

        self.previous_g_loss = 100
        self.previous_d_loss = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.genModel = self.build_generator()
        noise = Input(shape=(self.latent_dim,))
        mus = self.genModel(noise)
        self.generator = Model(noise, mus)

        self.critic = self.build_critic()

        # -------------------------------
        # Construct Computational Graph
        #       for the Critic
        # -------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Music input (real sample)
        real_mus = Input(shape=(576, 1))

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate music based of noise (fake sample)
        fake_mus = self.generator(z_disc)

        # Discriminator determines validity of the real and fake samples_TWGAN
        fake = self.critic(fake_mus)
        valid = self.critic(real_mus)

        # Construct weighted average between real and fake samples_TWGAN
        interpolated_mus = RandomWeightedAverage()([real_mus, fake_mus])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_mus)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_mus)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.critic_model = Model(inputs=[real_mus, z_disc],
                                  outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                  optimizer=optimizer,
                                  loss_weights=[1, 1, 10])
        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(100,))
        # Generate images based of noise
        mus = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(mus)

        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):

        """
        Computes gradient penalty based on prediction and weighted real / fake samples_TWGAN
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples_TWGAN
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):

        return K.mean(y_true * y_pred)

    # saves log info for graph plotting with tensorboard
    def write_log(self, callback, names, logs, epoch):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = logs
        summary_value.tag = names
        callback.writer.add_summary(summary, epoch)
        callback.writer.flush()

    def build_generator(self):

        model = Sequential()

        model.add(Dense(576, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((576, 1)))
        model.add(Conv1D(64, kernel_size=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv1D(32, kernel_size=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv1D(self.channels, kernel_size=1, padding="same"))
        model.add(Activation("tanh"))

        if SHOULD_LOAD_WEIGHTS:
            model.load_weights(LOAD_WEIGHTS_PATH)

        model.summary()

        return model

    def build_critic(self):

        model = compiled_tcn(num_feat=1,
                             num_classes=1,
                             nb_filters=32,
                             kernel_size=2,
                             dilations=[2 ** i for i in range(8)],
                             nb_stacks=2,
                             max_len=self.inp_cols,
                             activation='norm_relu',
                             use_skip_connections=True,
                             return_sequences=False,
                             regression=True)

        model.summary()

        mus = Input(shape=(576, 1))
        validity = model(mus)

        return Model(mus, validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        X_train = getData()

        print(np.shape(X_train))

        X_train = X_train.reshape(10, 576, 1)

        X_train = X_train.astype('float32')

        # Scaling the range of the datapoints to [-1, 1]
        # Because we are using tanh as the activation function in the last layer of the generator
        # and tanh restricts the weights_TWGAN in the range [-1, 1]
        X_train = (X_train - 22) / 22

        print(np.unique(X_train))

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))  # Dummy gt for gradient penalty

        # parameters for tensorboard logging of each model
        d_log_path = './d_logs'
        d_callback = TensorBoard(d_log_path)
        d_callback.set_model(self.critic_model)
        d_train_name = 'd_loss'

        g_log_path = './g_logs'
        g_callback = TensorBoard(g_log_path)
        g_callback.set_model(self.generator_model)
        g_train_name = 'g_loss'

        for epoch in range(epochs):

            for training_round in range(self.n_critic):

                # freeze training of critic when optimised
                if not self.previous_d_loss < 0.02:

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------

                    # Select a random batch of images
                    idx = np.random.randint(0, X_train.shape[0], batch_size)
                    inps = X_train[idx]
                    # Sample generator input
                    noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                    # Train the critic
                    d_loss = self.critic_model.train_on_batch([inps, noise],
                                                              [valid, fake, dummy])

                    # write tensorboard data for critic
                    if training_round == self.n_critic-1:
                        self.write_log(d_callback, d_train_name, d_loss[0], epoch)

                    # If g_loss improves then make samples_TWGAN
                    if abs(d_loss[0]) < self.previous_d_loss:
                        self.previous_d_loss = abs(d_loss[0])

            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
            # write tensorboard data for generator
            self.write_log(g_callback, g_train_name, g_loss, epoch)

            # If g_loss improves then make samples_TWGAN
            if abs(g_loss) < abs(self.previous_g_loss):
                # Plot the progress
                print("G improved, taking samples_TWGAN")
                self.previous_g_loss = g_loss
                self.save_samples(epoch)
                self.genModel.save_weights("weights_TWGAN/epoch_%d.h5" % epoch)

                # let the critic start training again if training had been frozen
                if self.previous_d_loss < 0.02:
                    self.previous_d_loss = 0.02

            # If at save interval => save generated image samples_TWGAN
            if (epoch - 1) % sample_interval == 0:
                self.save_samples(epoch)
                self.genModel.save_weights("weights_TWGAN/epoch_%d.h5" % epoch)
                if self.previous_d_loss < 0.02:
                    self.previous_d_loss = 0.02
                

    def save_samples(self, epoch):
        for i in range(15):
            noise = np.random.normal(0, 1, (1, self.latent_dim))
            gen_mus = self.generator.predict(noise)
            gen_mus = np.reshape(gen_mus, 576)
            gen_mus = fromCategorical(gen_mus)
            np.savetxt("samples_TWGAN/epoch_%d_%i.txt" % (epoch, i), gen_mus, fmt='%s')
            pruneNonCandidates()


if __name__ == '__main__':
    wgan = WGAN()
    wgan.train(epochs=5000, batch_size=BATCH_SIZE, sample_interval=SAMPLE_INTERVAL)
    

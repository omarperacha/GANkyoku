from __future__ import print_function, division
from utils import getData, fromCategorical, pruneNonCandidates, synthData, getSingleSample

from keras.layers import Input, Dense, Concatenate, Conv1D, LeakyReLU, Reshape
from keras.layers.merge import _Merge
from keras.layers import LSTM, Flatten, Dropout, CuDNNLSTM, BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.optimizers import RMSprop
from functools import partial
import tensorflow as tf

import random

import keras.backend as K

import numpy as np

BATCH_SIZE = 100
N_EPOCH = 100002
LOAD_WEIGHTS_PATH = "weights/epoch_17701.h5"
SHOULD_LOAD_WEIGHTS = False
SAMPLE_INTERVAL = 100
NUM_CONDS = 4


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
        self.latent_dim = 128

        self.previous_g_loss = 100
        self.previous_d_loss = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()

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
        z_cond = Input(shape=(NUM_CONDS,))
        # Generate music based of noise (fake sample)
        fake_mus = self.generator([z_disc, z_cond])

        # Discriminator determines validity of the real and fake samples
        fake = self.critic([fake_mus, z_cond])
        real_mus_cond = Input(shape=(NUM_CONDS,))
        valid = self.critic([real_mus, real_mus_cond])

        # Construct weighted average between real and fake samples
        interpolated_mus = RandomWeightedAverage()([real_mus, fake_mus])
        # Determine validity of weighted sample
        validity_interpolated = self.critic([interpolated_mus, z_cond])

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_mus)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.critic_model = Model(inputs=[real_mus, real_mus_cond, z_disc, z_cond],
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
        z_gen = Input(shape=(self.latent_dim,))
        zg_cond = Input(shape=(NUM_CONDS,))
        # Generate images based of noise
        mus = self.generator([z_gen, zg_cond])
        # Discriminator determines validity
        valid = self.critic([mus, zg_cond])

        # Defines generator model
        self.generator_model = Model([z_gen, zg_cond], valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):

        """
        Computes gradient penalty based on prediction and weighted real / fake samples
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
        gradient_penalty = 10 * K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
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

        noise = Input(shape=(self.latent_dim,))
        condition_tensor = Input(shape=(NUM_CONDS,))
        merged = Concatenate(axis=1)([noise, condition_tensor])

        model = Dense(576, activation="relu", input_dim=(self.latent_dim + NUM_CONDS))(merged)
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

    def build_critic(self):

        num_feat = 1
        max_len = self.inp_cols

        mus = Input(shape=(max_len, num_feat))
        condition_tensor = Input(shape=(NUM_CONDS,))

        model = Conv1D(16, kernel_size=2, padding="same")(mus)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.25)(model)
        model = Conv1D(32, kernel_size=2, padding="same")(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.25)(model)
        model = Conv1D(64, kernel_size=2, padding="same")(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.25)(model)
        model = Conv1D(128, kernel_size=2, padding="same")(model)
        model = BatchNormalization(momentum=0.8)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.25)(model)
        model = Flatten()(model)

        model = Concatenate(axis=1)([model, condition_tensor])
        model = Dense(1)(model)

        output_layer = model
        model = Model([mus, condition_tensor], output_layer)

        model.summary()

        validity = model([mus, condition_tensor])

        model = Model([mus, condition_tensor], validity)


        return model

    def train(self, epochs, batch_size=1, sample_interval=50):

        X_train, conds = self.getX()

        print(np.shape(X_train))

        # already normalised
        X_train = X_train.astype('float32')
        conds = conds.astype('float32')

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

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                inps = X_train[idx]
                inp_conds = conds[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                conds_fake = np.zeros((batch_size, NUM_CONDS))

                for i in range(batch_size):
                    cond = np.array([0, 0, 0, 0])
                    if i % 25 == 0:
                        cond[0] = 1
                    else:
                        randidx = random.randint(1, (NUM_CONDS - 1))
                        cond[randidx] = 1
                    conds_fake[i, :] = cond

                # Train the critic
                d_loss = self.critic_model.train_on_batch([inps, inp_conds, noise, conds_fake],
                                                          [valid, fake, dummy])

                # write tensorboard data for critic
                if training_round == self.n_critic - 1:
                    self.write_log(d_callback, d_train_name, d_loss[0], epoch)

                # If g_loss improves then make samples
                if abs(d_loss[0]) < self.previous_d_loss:
                    self.previous_d_loss = abs(d_loss[0])

            # ---------------------
            #  Train Generator
            # ---------------------
            g_loss = self.generator_model.train_on_batch([noise, conds_fake], valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
            # write tensorboard data for generator
            self.write_log(g_callback, g_train_name, g_loss, epoch)

            # If g_loss improves then make samples
            if abs(g_loss) < abs(self.previous_g_loss):
                # Plot the progress
                print("G improved, taking samples")
                self.previous_g_loss = g_loss
                self.save_samples(epoch)
                self.generator.save_weights("weights/epoch_%d.h5" % epoch)

            # If at save interval => save generated image samples
            if (epoch - 1) % sample_interval == 0:
                self.save_samples(epoch)
                self.generator.save_weights("weights/epoch_%d.h5" % epoch)

    def save_samples(self, epoch):
        for i in range(4):
            noise = np.random.normal(0, 1, (1, self.latent_dim))
            true_class = np.zeros((1, NUM_CONDS))
            true_class[0, 0] = 1
            gen_mus = self.generator.predict([noise, true_class])
            gen_mus = np.reshape(gen_mus, 576)
            gen_mus = fromCategorical(gen_mus)
            np.savetxt("samples/epoch_%d_%i.txt" % (epoch, i), gen_mus, fmt='%s')
        pruneNonCandidates()

    def getX(self):
        samples = getData()
        x_train = np.zeros((BATCH_SIZE, 576))
        conds = np.zeros((BATCH_SIZE, NUM_CONDS))
        for i in range(BATCH_SIZE):
            if i % 25 == 0:
                x = getSingleSample(samples)
                cond = np.array([1, 0, 0, 0])

            else:
                x = synthData((i % 25) / 25, samples)

                if (i % 25) > 17:
                    cond = np.array([0, 0, 0, 1])
                elif (i % 25) > 8:
                    cond = np.array([0, 0, 1, 0])
                else:
                    cond = np.array([0, 1, 0, 0])

            x_train[i, :] = x
            conds[i, :] = cond

        x_train = np.reshape(x_train, (BATCH_SIZE, 576, 1))

        return x_train, conds


if __name__ == '__main__':
    wgan = WGAN()
    wgan.train(epochs=N_EPOCH, batch_size=BATCH_SIZE, sample_interval=SAMPLE_INTERVAL)
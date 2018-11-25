'''
Created on 18 Nov 2018

@author: omarperacha
'''

from __future__ import print_function, division
from utils import getData, fromCategorical, pruneNonCandidates

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.merge import _Merge
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv1D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial

import keras.backend as K


import numpy as np

BATCH_SIZE = 20
LOAD_WEIGHTS_PATH = "weights/epoch_0.h5"
SHOULD_LOAD_WEIGHTS = False

class RandomWeightedAverage(_Merge):
    
    """Provides a (random) weighted average between real and generated image samples"""
    
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
        self.n_critic = 3
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.genModel = self.build_generator()
        noise = Input(shape=(self.latent_dim,))
        mus = self.genModel(noise)
        self.generator = Model(noise, mus)

        self.critic = self.build_critic()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Music input (real sample)
        real_mus = Input(shape=(576, 1))

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate music based of noise (fake sample)
        fake_mus = self.generator(z_disc)

        # Discriminator determines validity of the real and fake samples
        fake = self.critic(fake_mus)
        valid = self.critic(real_mus)

        # Construct weighted average between real and fake samples
        interpolated_mus = RandomWeightedAverage()([real_mus, fake_mus])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_mus)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_mus)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_mus, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

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
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        
        return K.mean(y_true * y_pred)
    
    

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

        if SHOULD_LOAD_WEIGHTS:
            model.load_weights(LOAD_WEIGHTS_PATH)

        model.summary()

        return model

    def build_critic(self):

        model = Sequential()

        model.add(Conv1D(32, kernel_size=1, padding="same", input_shape=(576, 1)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(64, kernel_size=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(128, kernel_size=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(256, kernel_size=4, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

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
        # and tanh restricts the weights in the range [-1, 1]
        X_train = (X_train - 22) / 22

        print(np.unique(X_train))

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                #freeze training of critic when optimised
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

                    # If g_loss improves then make samples
                    if abs(d_loss[0]) < self.previous_d_loss:
                        self.previous_d_loss = abs(d_loss[0])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # If g_loss improves then make samples
            if abs(g_loss) < abs(self.previous_g_loss):
                # Plot the progress
                print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
                self.previous_g_loss = g_loss
                self.save_samples(epoch)
                self.genModel.save_weights("weights/epoch_%d.h5" % epoch)


            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # Plot the progress
                print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
                self.save_samples(epoch)
                self.genModel.save_weights("weights/epoch_%d.h5" % epoch)
                

    def save_samples(self, epoch):
        for i in range(15):
            noise = np.random.normal(0, 1, (1, self.latent_dim))
            gen_mus = self.generator.predict(noise)
            gen_mus = np.reshape(gen_mus, (576))
            gen_mus = fromCategorical(gen_mus)
            np.savetxt("samples/epoch_%d_%i.txt" % (epoch, i), gen_mus, fmt='%s')




if __name__ == '__main__':
    wgan = WGAN()
    wgan.train(epochs=3000, batch_size=BATCH_SIZE, sample_interval=100)
    pruneNonCandidates()
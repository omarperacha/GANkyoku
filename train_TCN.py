'''
Created on 30 Nov 2018

@author: omarperacha
'''
import numpy as np
from utils import getDataVariedLength, oneHotEncode, getTotalSteps
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import Dense
from keras.models import Input, Model
from keras.optimizers import Adam
from tcn import TCN

# CHANGE PATH AND FILETYPE IN glob.glob TO MATCH YOUR TRAINING DATA
samples = getDataVariedLength()
array_length = len(samples)
NUM_CLASSES = 45
BATCH_SIZE = getTotalSteps()
NUM_EPOCHS = 3000

def getXY():
    for i in range(NUM_EPOCHS):
        n_patterns = 0
        for i in range(array_length):
            seq_length = 5
            data = np.array(samples[i])
            n_tokens = len(data)
            # prepare X & y data
            for i in range(0, n_tokens - seq_length, 1):
                seq_in = data[0:seq_length]
                # normalize
                seq_in = seq_in / float(45)
                # reshape X to be [samples, time steps, features]
                seq_in = np.reshape(seq_in, (1, seq_length, 1))
                seq_out = data[seq_length]
                seq_out = oneHotEncode(seq_out, NUM_CLASSES)
                seq_out = np.reshape(seq_out, (1, 1, NUM_CLASSES))
                seq_length += 1
                n_patterns += 1
                yield seq_in, seq_out


# define the TCN model
i = Input(batch_shape=(1, None, 1))
model = TCN(nb_filters = 64,
             kernel_size = 1, 
             nb_stacks = 1, 
             dilations = [2 ** i for i in range(8)], 
             activation = 'norm_relu', 
             use_skip_connections = True, 
             dropout_rate = 0.05, 
             return_sequences = True)(i)
model = Dense(NUM_CLASSES, activation='softmax')(model)

model = Model(inputs=[i], outputs=[model])

model.summary()
# UNCOMMENT NEXT TWO LINES TO LOAD YUOR OWN WEIGHTS (OR THE ONES PROVIDED)
# filename = "weights-improvement-00-0.4125-3.hdf"
# model.load_weights(filename)

# CHANGE lr TO ADJUST LEARNING RATE AS YOU DESIRE. (A DECAYING RATE WORKS WELL).
adam = Adam(lr=0.02, clipnorm=1.)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# checkpoint after each training epoch - weights saved only if loss improves
filepath = "weights_TCN/{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=15, min_lr=0.0005)
log = TensorBoard(log_dir='./logs_tcn')
callbacks_list = [checkpoint, reduce_lr, log]

gen = getXY()
# fit the model
model.fit_generator(gen, epochs=NUM_EPOCHS, steps_per_epoch=BATCH_SIZE, max_queue_size=1, callbacks=callbacks_list, verbose=2)


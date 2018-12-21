import numpy as np
from utils import getDataVariedLength, oneHotEncodeLSTM, getTotalSteps, getSingleSample, synthData
from keras.models import Input, Model
from keras.layers import Dense
from keras.layers import Dropout, Concatenate
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from keras.utils import np_utils
from keras.optimizers import Adam
import random
# CHANGE PATH AND FILETYPE IN glob.glob TO MATCH YOUR TRAINING DATA
samples = getDataVariedLength()
array_length = len(samples)

NUM_CLASSES = 45
NUM_EPOCHS = 30000
SEQ_LENGTH = 6
BATCH_SIZE = getTotalSteps()
SHOULD_RESET_STATES = False
print(BATCH_SIZE)

class stateReset(Callback):
    def __init__(self):
        self.should_reset = False

    def on_batch_end(self, batch, logs={}):
        if self.should_reset:
            self.model.reset_states()
            self.should_reset = False

resetStates = stateReset()

def getXY():
    for e in range(NUM_EPOCHS):
        n_patterns = 0

        for i in range(array_length):
            count = 0
            choice = random.randint(0,9)
            resetStates.should_reset = True
            if choice == 1:
                data = np.array(getSingleSample(samples, False, i))
                cond = [1,0,0,0]
            elif choice < 4:
                data = np.array(synthData((choice / 10), samples, False, i))
                cond = [0, 1, 0, 0]
            elif choice < 7:
                data = np.array(synthData((choice / 10), samples, False, i))
                cond = [0, 0, 1, 0]
            elif choice < 10:
                data = np.array(synthData((choice / 10), samples, False, i))
                cond = [0, 0, 0, 1]
            n_tokens = len(data)
            cond = np.array(cond)
            cond = np.reshape(cond, newshape=(1, 4))
            # prepare X & y data
            for i in range(0, n_tokens - SEQ_LENGTH, 1):
                seq_in = data[count:(SEQ_LENGTH+count)]
                # reshape X to be [samples, time steps, features]
                seq_in = np.reshape(seq_in, (1, SEQ_LENGTH, 1))
                seq_out = data[SEQ_LENGTH + count]
                seq_out = oneHotEncodeLSTM(seq_out, NUM_CLASSES)
                seq_out = np.reshape(seq_out, (1, NUM_CLASSES))
                count += 1
                n_patterns += 1
                yield [seq_in, cond], seq_out


# define the LSTM model
i = Input(batch_shape=(1, SEQ_LENGTH, 1))
c = Input(batch_shape=(1, 4))
model = LSTM(128, input_shape=(None, 1), return_sequences=True, stateful=True)(i)
model = Dropout(0.2)(model)
model = LSTM(128, return_sequences=False)(model)
model = Dropout(0.2)(model)
model = Concatenate(1)([model, c])
model = Dense(64)(model)
model = Dense(NUM_CLASSES, activation='softmax')(model)
model = Model(inputs=[i, c], outputs=[model])

# UNCOMMENT NEXT TWO LINES TO LOAD YUOR OWN WEIGHTS (OR THE ONES PROVIDED)
# filename = "weights-improvement-00-0.4125-3.hdf"
# model.load_weights(filename)
adam = Adam(amsgrad=True)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

# checkpoint after each training epoch - weights saved only if loss improves
filepath = "weights_LSTM/{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
log = TensorBoard(log_dir='./logs_LSTM')

callbacks_list = [checkpoint, log, resetStates]

gen = getXY()
# fit the model
model.fit_generator(gen, epochs=NUM_EPOCHS, steps_per_epoch=BATCH_SIZE, max_queue_size=1, callbacks=callbacks_list, verbose=2)

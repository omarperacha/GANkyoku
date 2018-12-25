import numpy as np
from utils import getDataVariedLength, oneHotEncodeLSTM, getTotalSteps, getSingleSample, synthData
from keras.models import Input, Model
from keras.layers import Dense, LeakyReLU
from keras.layers import Dropout, Concatenate
from keras.layers import CuDNNLSTM, LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, Callback
from keras.optimizers import Adam
import random
# CHANGE PATH AND FILETYPE IN glob.glob TO MATCH YOUR TRAINING DATA
samples = getDataVariedLength()
array_length = len(samples)
print(array_length)

NUM_CLASSES = 45
NUM_EPOCHS = 1000
N_BATCHES = 300
SEQ_LENGTH = 6
BATCH_SIZE = getTotalSteps()
print(BATCH_SIZE)


def getXY():
    n_patterns = 0

    X = []
    Conds = []
    Y = []

    for _ in range(N_BATCHES):

        for i in range(array_length):
            count = 0
            choice = random.randint(0, 9)
            if choice == 0:
                data = np.array(getSingleSample(samples, False, i))
                cond = [1, 0, 0, 0]
            elif choice < 3:
                data = np.array(synthData((choice / 10), samples, False, i))
                cond = [0, 1, 0, 0]
            elif choice < 7:
                data = np.array(synthData((choice / 10), samples, False, i))
                cond = [0, 0, 1, 0]
            else:
                data = np.array(synthData((choice / 10), samples, False, i))
                cond = [0, 0, 0, 1]

            #renormalise
            data = ((data*22)+22)/45

            n_tokens = len(data)

            # prepare X & y data
            for i in range(0, n_tokens - SEQ_LENGTH, 1):
                seq_in = data[count:(SEQ_LENGTH+count)]
                X.append(seq_in)

                Conds.append(cond)

                seq_out = data[SEQ_LENGTH + count]
                seq_out = oneHotEncodeLSTM(seq_out, NUM_CLASSES)
                Y.append(seq_out)

                count += 1
                n_patterns += 1

    X = np.array(X)
    X = np.reshape(X, (n_patterns, SEQ_LENGTH, 1))

    Conds = np.array(Conds)
    Conds = np.reshape(Conds, (n_patterns, 4))

    Y = np.array(Y)
    Y = np.reshape(Y, (n_patterns, NUM_CLASSES))

    return X, Conds, Y


# define the LSTM model
i = Input((SEQ_LENGTH, 1))
c = Input((4, ))
model = CuDNNLSTM(128, input_shape=(None, 1), return_sequences=True, stateful=False)(i)
model = Dropout(0.2)(model)
model = CuDNNLSTM(128, return_sequences=False)(model)
model = Dropout(0.2)(model)
model = Concatenate(1)([model, c])
model = Dense(NUM_CLASSES, activation='softmax')(model)
model = Model(inputs=[i, c], outputs=[model])

# UNCOMMENT NEXT TWO LINES TO LOAD YUOR OWN WEIGHTS (OR THE ONES PROVIDED)
#filename = "weights-improvement-00-0.4125-3.hdf"
#model.load_weights(filename)
adam = Adam(amsgrad=True)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

# checkpoint after each training epoch - weights saved only if loss improves
filepath = "weights_LSTM/{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
#log = TensorBoard(log_dir='./logs_LSTM')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, min_lr=0.0005)

callbacks_list = [checkpoint]

X, Conds, Y = getXY()
model.fit([X, Conds], Y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1, callbacks=callbacks_list)

import numpy as np
from utils import getData
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import np_utils
from keras.optimizers import RMSprop

# CHANGE PATH AND FILETYPE IN glob.glob TO MATCH YOUR TRAINING DATA
samples = getData()
array_length = len(samples)

# CHANGE THESE VALUES TO MATCH YOUR TRAINING IMAGES
sample_width = 1
sample_height = 576

n_tokens = sample_width * sample_height

# CHANGE seq_length TO ANOTHER VALUE IF DESIRED
seq_length = 5
dataX = []
dataY = []
# convert each image's pixel values into a column vector
for i in range(array_length):
    data = samples[i, :]
    # prepare X & y data
    for i in range(0, n_tokens - seq_length, 1):
        seq_in = data[i:i + seq_length]
        seq_out = data[i + seq_length]
        dataX.append(seq_in)
        dataY.append(seq_out)
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(45)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)


# define the LSTM model
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

# UNCOMMENT NEXT TWO LINES TO LOAD YUOR OWN WEIGHTS (OR THE ONES PROVIDED)
# filename = "weights-improvement-00-0.4125-3.hdf"
# model.load_weights(filename)

# CHANGE lr TO ADJUST LEARNING RATE AS YOU DESIRE. (A DECAYING RATE WORKS WELL).
rms = RMSprop(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=rms)

# checkpoint after each training epoch - weights saved only if loss improves
filepath = "weights_LSTM/{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=5, min_lr=0.0005)
callbacks_list = [checkpoint]

# fit the model
model.fit(X, y, epochs=2000, batch_size=571, callbacks=callbacks_list)
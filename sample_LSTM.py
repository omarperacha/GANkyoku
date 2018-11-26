import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout, Lambda
from keras.layers import LSTM
from utils import getData, fromCategoricalNoScaling, pruneNonCandidates
import datetime


# CHANGE PATH AND FILETYPE IN glob.glob TO MATCH YOUR TRAINING DATA
samples = getData()
array_length = len(samples)
num_classes = 45
temp = 0.6

# CHANGE THESE VALUES TO MATCH YOUR TRAINING IMAGES
sample_width = 1
sample_height = 576

n_tokens = sample_width * sample_height

# CHANGE seq_length TO ANOTHER VALUE IF DESIRED
seq_length = 5
dataX = []
# convert each image's pixel values into a column vector
for i in range(array_length):
    data = samples[i, 0:seq_length]
    dataX.append(data)
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(num_classes)


# define the LSTM model
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Lambda(lambda x: x / temp))
model.add(Activation('softmax'))

#CHANGE filename TO LOAD YOUR OWN WEIGHTS
filename = "weights_LSTM/315-0.2483.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
pattern_temp = pattern
print ("starting")


# generate image
for i in range(n_tokens-seq_length):
    x = np.reshape(pattern_temp, (1, len(pattern_temp), 1))
    x = x / float(num_classes)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    print(i, index)
    pattern = np.append(pattern, index)
    pattern_temp = pattern[(1+i):len(pattern)]
print("Done.")
pattern = fromCategoricalNoScaling(pattern)
print(pattern)
np.savetxt("samples_LSTM/%s.txt" % datetime.datetime.now(), pattern, fmt='%s')
pruneNonCandidates()


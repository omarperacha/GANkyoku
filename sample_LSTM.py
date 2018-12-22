import numpy as np
from keras.models import Input, Model
from keras.layers import Dense
from keras.layers import Dropout, Concatenate
from keras.layers import LSTM, LeakyReLU
from keras.optimizers import Adam
from utils import getData, fromCategoricalNoScaling, toCategoricalAlreadyDict
import datetime


# CHANGE PATH AND FILETYPE IN glob.glob TO MATCH YOUR TRAINING DATA
samples = getData()
array_length = len(samples)
num_classes = 45
temp = 1.0
SEQ_LENGTH = 6

# CHANGE THESE VALUES TO MATCH YOUR TRAINING IMAGES
sample_width = 1
sample_height = 576

n_tokens = sample_width * sample_height

# CHANGE seq_length TO ANOTHER VALUE IF DESIRED
seq_length = SEQ_LENGTH
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
# define the LSTM model
i = Input(batch_shape=(1, SEQ_LENGTH, 1))
c = Input(batch_shape=(1, 4))
model = LSTM(128, input_shape=(None, 1), return_sequences=True, stateful=True)(i)
model = LeakyReLU()(model)
model = Dropout(0.2)(model)
model = LSTM(128, return_sequences=False)(model)
model = LeakyReLU()(model)
model = Dropout(0.2)(model)
model = Concatenate(1)([model, c])
model = Dense(64)(model)
model = LeakyReLU()(model)
model = Dense(num_classes, activation='softmax')(model)
model = Model(inputs=[i, c], outputs=[model])

#CHANGE filename TO LOAD YOUR OWN WEIGHTS
filename = "weights_LSTM/122-0.5177.hdf5"
model.load_weights(filename)
adam = Adam(lr=0.00005)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

# pick a random seed
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
#pattern = toCategoricalAlreadyDict(pattern)
pattern_temp = pattern
print ("starting")

cond = np.reshape(np.array([1, 0, 0, 0]), newshape=(1, 4))

# generate image
for i in range(n_tokens-seq_length):
    x = np.reshape(pattern_temp, (1, len(pattern_temp), 1))
    prediction = model.predict([x, cond], verbose=0)
    index = np.argmax(prediction)
    print(i, index)
    pattern = np.append(pattern, index)
    pattern_temp = pattern[(1+i):len(pattern)]
print("Done.")
pattern = fromCategoricalNoScaling(pattern)
print(pattern)
np.savetxt("samples_LSTM/%s.txt" % datetime.datetime.now(), pattern, fmt='%s')


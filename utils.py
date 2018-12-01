import glob
import os
import numpy as np
import random

labelDict = {}
NUM_CLASSES = 45


def getData():
    vectorisedSamples = glob.glob("vectorised dataset/*.csv")
    
    allData = np.full((10, 576), 'END', dtype='object')
    
    count = 0
    for i in vectorisedSamples:
        npData = np.full((576,), 'END', dtype='object')
        data = np.genfromtxt(i, delimiter=',', dtype='str')
        l = len(data)
        npData[0:l,] = data
        print(i, ": ", l)
        allData[count, 0:] = npData
        count += 1
        
    allData = toCategorical(allData)
    
    return allData

def getDataLSTMTrain():
    vectorisedSamples = glob.glob("vectorised dataset/*.csv")
    
    l = getTotalFeatureCount()
    mul = 10
    
    allData = np.full((l*mul), 'END', dtype='object')
    
    el = 0
    for _ in range(mul):
        aux = vectorisedSamples
        random.shuffle(aux)
        
        for i in aux:
            data = np.genfromtxt(i, delimiter=',', dtype='str')
            l = len(data)
            l2 = l+9
            allData[el:l+el] = data
            print(i, ": ", l)
            el+=l2
        
    allData = toCategorical(allData)
    
    return allData

def getDataUncategorised():
    vectorisedSamples = glob.glob("vectorised dataset/*.csv")
    
    allData = np.full((10, 576), 'END', dtype='object')
    
    count = 0
    for i in vectorisedSamples:
        npData = np.full((576,), 'END', dtype='object')
        data = np.genfromtxt(i, delimiter=',', dtype='str')
        l = len(data)
        npData[0:l,] = data
        print(i, ": ", l)
        allData[count, 0:] = npData
        count += 1
        
    return allData
        

def getDataVariedLength():
    vectorisedSamples = glob.glob("vectorised dataset/*.csv")
    
    allData = []
    
    count = 0
    for i in vectorisedSamples:
        data = np.genfromtxt(i, delimiter=',', dtype='str')
        l = len(data)
        print(i, ": ", l)
        allData.append(data)
        count += 1
        
    allData = toCategoricalVariedLength(allData)
    
    return allData


def toCategorical(myArray):
    w = myArray.shape[0]
    h = None
    
    if len(myArray.shape) > 1:
        h = myArray.shape[1]
        
    size = myArray.flatten().shape[0]
    print(size)
    newArray = np.ones(size)
    myArray = np.reshape(myArray, (size))
    unique = np.unique(myArray)
    for i in range(NUM_CLASSES):
        labelDict[i] = unique[i]

    for j in range(size):
        newArray[j] = unique.tolist().index(myArray[j])

    if h != None:
        newArray = np.reshape(newArray, (w, h))

    return newArray

def toCategoricalAlreadyDict(myArray):
    w = myArray.shape[0]
    h = None
    
    if len(myArray.shape) > 1:
        h = myArray.shape[1]
        
    size = myArray.flatten().shape[0]
    print(size)
    newArray = np.ones(size)
    myArray = np.reshape(myArray, (size))

    for j in range(size):
        
        newArray[j] = list(labelDict.keys())[list(labelDict.values()).index(myArray[j])]
        print(list(labelDict.keys())[list(labelDict.values()).index(myArray[j])])

    if h != None:
        newArray = np.reshape(newArray, (w, h))

    return newArray


def toCategoricalVariedLength(myArray):
    unique = np.unique(np.reshape(getDataUncategorised(),(5760)))
    for i in range(45):
        labelDict[i] = unique[i]
    
    samples = [[],[],[],[],[],[],[],[],[],[]]
    
    count = 0
    for piece in myArray:
        newArray = []
        for j in range(len(piece)):
            newArray.append((unique.tolist().index(piece[j])))
        samples[count] = newArray
        count+=1

    return samples

def oneHotEncode(myVal, numClasses):
    y = np.zeros(numClasses)
    y[myVal] = 1
    return y


def fromCategorical(myArray):
    retransformed = np.full((576), 'END', dtype='object')
    myArray = myArray * 22
    myArray = myArray + 22
    myArray = np.rint(myArray)
    for i in range(576):
        retransformed[i] = labelDict[myArray[i]]

    return retransformed

def pruneNonCandidates():
    samples = glob.glob("samples_LSTM/*.txt")
    for sample in samples:
        data = np.genfromtxt(sample, delimiter=',', dtype='str')
        if data[575] != 'END' or data[160] == 'END':
            os.remove(sample)

def fromCategoricalNoScaling(myArray):
    retransformed = np.full((576), 'END', dtype='object')
    myArray = np.rint(myArray)
    for i in range(576):
        retransformed[i] = labelDict[myArray[i]]

    return retransformed

def getTotalSteps():
    samples=getDataVariedLength()
    n_patterns = 0
    for i in range(10):
        seq_length = 5
        data = np.array(samples[i])
        n_tokens = len(data) + 9
        # prepare X & y data
        for i in range(0, n_tokens - seq_length, 1):
            seq_length += 1
            n_patterns += 1
    return n_patterns

def getTotalFeatureCount():
    vectorisedSamples = glob.glob("vectorised dataset/*.csv")
    count = 0
    
    for sample in vectorisedSamples:
        data = np.genfromtxt(sample, delimiter=',', dtype='str')
        count+=len(data)
        count+=9
        
    return count


getDataLSTMTrain()


import glob
import os
import numpy as np

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
    newArray = np.ones(5760)
    myArray = np.reshape(myArray, (5760))
    unique = np.unique(myArray)
    for i in range(45):
        labelDict[i] = unique[i]

    for j in range(5760):
        newArray[j] = unique.tolist().index(myArray[j])

    newArray = np.reshape(newArray, (10, 576))

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
        n_tokens = len(data)
        # prepare X & y data
        for i in range(0, n_tokens - seq_length, 1):
            seq_length += 1
            n_patterns += 1
    return n_patterns


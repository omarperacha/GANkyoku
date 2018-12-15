import glob
import os
import numpy as np
import random

labelDict = {0:'START', 1:'otsu_symbol', 2:'kan_symbol', 3:'hi_gracenote', 4:'u_gracenote',
             5:'da_three', 6:'san_no_chi_meri', 7:'shi_ni_go_no_ha', 8:'go_no_ha', 9:'go_no_hi', 10:'hi',
             11:'hi_meri', 12:'ha', 13:'a', 14:'ho',
             15:'ra', 16:'karakara', 17:'horohoro', 18:'korokoro', 19:'trill', 
             20:'yuri', 21:'nayashi', 22:'-', 23:'meri', 24:'tsu',
             25:'tsu_meri', 26:'re', 27:'u', 28:'u_meri', 29:'ru', 
             30:'ru_meri', 31:'re_meri', 32:'chi', 33:'chi_meri' , 34:'ri_meri',
             35:'ri', 36:'i', 37:'ro', 38:'ro_dai_meri' , 39:'ro_meri', 
             40:'ro_dai_kan', 41:'san_no_u', 42:'san_no_u_meri', 43:'suriage', 44:'END'}
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

    for j in range(size):
        newArray[j] = list(labelDict.keys())[list(labelDict.values()).index(myArray[j])]

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

def oneHotEncode(myVal, numClasses=45):
    y = np.zeros(numClasses)
    y[myVal] = 1
    return y


def oneHotEncodeSequence(myArray, numClasses=45):

    outputArr = np.zeros((576, numClasses))

    for i in range(len(myArray)):
        outputArr[i, :] = oneHotEncode(int(myArray[i]), numClasses)

    return(outputArr)


def fromCategorical(myArray):
    retransformed = np.full((576), 'END', dtype='object')
    myArray = myArray * 22
    myArray = myArray + 22
    myArray = np.rint(myArray)
    for i in range(576):
        retransformed[i] = labelDict[myArray[i]]

    return retransformed

def pruneNonCandidates():
    samples = glob.glob("samples_TWGAN/*.txt")
    for sample in samples:
        data = np.genfromtxt(sample, delimiter=',', dtype='str')
        if data[0] != 'START':
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


def synthData(noiseFactor, data):

    randIdx = random.randint(0,9)
    #print(randIdx)
    sample = data[randIdx]
    sample = (sample -22)/22
    for i in range(len(sample)):
        if abs(sample[i]) != 1:
            sample[i] = sample[i] * random.uniform(1-noiseFactor, 1+noiseFactor)
            sample[i] = np.tanh(sample[i])
    return sample

def getSingleSample(data):

    randIdx = random.randint(0,9)
    sample = data[randIdx]
    sample = (sample - 22)/22

    return sample

def vetCWGANoutputs():
    threshold = 25
    dir = glob.glob("samples_TWGAN/*.txt")
    for output in dir:
        count = 0
        resetCount = 0
        with open(output, 'r') as file:
            for word in file:
                count += 1
                if '-' in word:
                    count = 0
                    resetCount += 1
                if count == threshold:
                    break
                if 'END' in word:
                    break
        if count < threshold and resetCount > 2:
            print("survivor", output)

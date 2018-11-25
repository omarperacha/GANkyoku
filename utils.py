import glob
import os
import numpy as np

labelDict = {}

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


def toCategorical(myArray):
    newArray = np.ones(5760)
    myArray = np.reshape(myArray, (5760))
    unique = np.unique(myArray)
    for i in range(45):
        labelDict[i] = unique[i]

    for j in range(5760):
        newArray[j] = unique.tolist().index(myArray[j])

    return newArray



def fromCategorical(myArray):
    retransformed = np.full((576), 'END', dtype='object')
    myArray = myArray * 22
    myArray = myArray + 22
    myArray = np.rint(myArray)
    for i in range(576):
        retransformed[i] = labelDict[myArray[i]]

    return retransformed

def pruneNonCandidates():
    samples = glob.glob("samples/*.txt")
    for sample in samples:
        data = np.genfromtxt(sample, delimiter=',', dtype='str')
        if data[0] != 'START':
            os.remove(sample)


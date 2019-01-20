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

    vectorisedSamples = glob.glob("PH_Shaku dataset/*.csv")
    
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


def fromCategorical(myArray):

    retransformed = np.full(576, 'END', dtype='object')
    myArray = myArray * 22
    myArray = myArray + 22
    myArray = np.rint(myArray)
    for i in range(576):
        retransformed[i] = labelDict[myArray[i]]

    return retransformed


def synthData(noiseFactor, data, rand=True, idx=0):

    if rand:
        randIdx = random.randint(0,9)
        sample = data[randIdx].copy()
    else:
        sample = data[idx].copy()

    for i in range(len(sample)):
        sample[i] = (sample[i] - 22)/22
        if abs(sample[i]) != 1:
            sample[i] = sample[i] * random.uniform(1-noiseFactor, 1+noiseFactor)
            sample[i] = np.tanh(sample[i])
        if not rand:
            sample[i] = ((int((sample[i]*22)+22))-22)/22
    return sample

def getSingleSample(data, rand=True, idx=0):

    if rand:
        randIdx = random.randint(0,9)
        sample = data[randIdx].copy()
    else:
        sample = data[idx].copy()

    for i in range(len(sample)):
        sample[i] = (sample[i] - 22)/22

    return sample


def pruneNonCandidates():
    samples = glob.glob("samples/*.txt")
    for sample in samples:
        data = np.genfromtxt(sample, delimiter=',', dtype='str')
        if data[0] != 'START':
            os.remove(sample)


def vetCWGANoutputs():
    threshold = 21
    dir = glob.glob("samples/*.txt")
    survivor_count = 0
    survivors = []
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
        if count < threshold and resetCount > 4:
            survivors.append(output)
            survivor_count += 1
        #else:
            #os.remove(output)
    survivors = sorted(survivors)
    for survivor in survivors:
        print(survivor)
    print("total survivors: ", survivor_count)



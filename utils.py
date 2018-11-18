import glob
import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder


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
    
    return(allData)


def toCategorical(myArray):
    myArray = np.reshape(myArray, (5760))
    encoder = LabelEncoder()
    transfomed_label = encoder.fit_transform(myArray)
    transfomed_label = np.reshape(transfomed_label, (10, 576))
    return(transfomed_label)




import glob
import csv
import numpy as np


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
        
    print(np.shape(allData))
    print(np.unique(allData))
    return(allData)

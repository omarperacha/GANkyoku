import glob
import csv
import numpy as np

vectorisedSamples = glob.glob("vectorised dataset/*.csv")

allData = np.array([], dtype='str')


for i in vectorisedSamples:
    data = np.genfromtxt(i, delimiter=',', dtype='str')
    print(i, ": ", len(data))
    allData = np.append(allData, data)
    
print(np.shape(allData))
print(np.unique(allData))

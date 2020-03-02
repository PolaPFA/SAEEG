import  csv
from collections import defaultdict
import numpy as np
import pandas
import pywt
from matplotlib import mlab
H =1
fout_data = open("train.csv",'a')
mypath = "E:\\College\\Graduation Project\\Dataset\\DEAP Dataset\\data_preprocessed_python\\data\\"

out = []

for i in range(32):
    if i < 10:
        name = '%0*d' % (2,i+1)
    else:
        name = i+1
    fnameReal = mypath+'convertedData\\s'+str(name)+".csv"
    with open(fnameReal) as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        c = 1
        temp2 = []
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            if c % 33 == 0:
                out.append(temp2)
                temp2 = []
                c += 1
                continue
            temp = list(row.values())
            temp = [float(i) for i in temp]
            coeffs, freq = mlab.psd(temp, Fs=128)
            temp2.extend(list(coeffs))
            if c == 1319:
                out.append(temp2)
                temp2 = []
            c += 1
        print(H)
        H += 1


out_data = pandas.DataFrame(out, dtype=float)
out_data.to_csv(mypath + 'featuresPSD.csv', mode='w', index=False)

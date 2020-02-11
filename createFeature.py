import  csv
from collections import defaultdict
import numpy as np
import pandas
import pywt
H =1
fout_data = open("train.csv",'a')
mypath = "E:/4 Year/GP/Dataset/DEAP Dataset/data_preprocessed_python/data_preprocessed_python/"

out = []
out2=[]
out3=[]
for i in range(32):
    if i < 10:
        name = '%0*d' % (2,i+1)
    else:
        name = i+1
    fnameReal = mypath+'convertedData/s'+str(name)+".csv"
    fnameFast = mypath+'interpolatesData\\f'+str(name)+".csv"
    fnameSlow = mypath+'interpolatesData\\s'+str(name)+".csv"
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
            coeffs = pywt.wavedec(temp, 'db4', level=6)
            cA6, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
            cD5 = np.std(cD5)
            cD4 = np.std(cD4)
            cD3 = np.std(cD3)
            cD2 = np.std(cD2)
            cD1 = np.std(cD1)
            temp2.append(cD1)
            temp2.append(cD2)
            temp2.append(cD3)
            temp2.append(cD4)
            temp2.append(cD5)
            if c == 1319:
                out.append(temp2)
                temp2 = []
            c += 1
        print(H)
        H += 1

    with open(fnameFast) as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        c = 1
        temp2 = []
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            if c % 33 == 0:
                out.append(temp2)
                temp2 = []
                c += 1
                continue
            temp =list(row.values())
            temp = [float(i) for i in temp]
            coeffs = pywt.wavedec(temp, 'db4', level=6)
            cA6, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
            cD5 = np.std(cD5)
            cD4 = np.std(cD4)
            cD3 = np.std(cD3)
            cD2 = np.std(cD2)
            cD1 = np.std(cD1)
            temp2.append(cD1)
            temp2.append(cD2)
            temp2.append(cD3)
            temp2.append(cD4)
            temp2.append(cD5)
            if c == 1319:
                out2.append(temp2)
                temp2 = []
            c += 1
    with open(fnameSlow) as f:
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
            coeffs = pywt.wavedec(temp, 'db4', level=6)
            cA6, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
            cD5 = np.std(cD5)
            cD4 = np.std(cD4)
            cD3 = np.std(cD3)
            cD2 = np.std(cD2)
            cD1 = np.std(cD1)
            temp2.append(cD1)
            temp2.append(cD2)
            temp2.append(cD3)
            temp2.append(cD4)
            temp2.append(cD5)
            if c == 1319:
                out3.append(temp2)
                temp2 = []
            c += 1
out_data = pandas.DataFrame(out)
out_data.to_csv(mypath + 'features.csv', mode='w', index=False)
out_data2 = pandas.DataFrame(out2)
out_data2.to_csv(mypath + 'features.csv', mode='a', index=False)
out_data3 = pandas.DataFrame(out3)
out_data3.to_csv(mypath + 'features.csv', mode='a', index=False)

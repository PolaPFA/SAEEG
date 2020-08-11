import csv
import numpy as np
import pandas
import pywt
H =1

mypath = "G:\\Datasets&GP\\DEAP\\data_preprocessed_python\\convertedData\\Allelectrodes\\"


data = []
for i in range(1,33 ):
    name = ""
    if i < 10:
        name = mypath + "s0" + str(i) + ".csv"
    else:
        name = mypath + "s" + str(i) + ".csv"
    print(name)
    with open(name) as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        itr = 1
        temprow = []
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            temp = list(row.values())
            temp = [float(i) for i in temp]
            coeffs = pywt.wavedec(temp, 'db4', level=6)
            cA6, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
            cD5 = np.std(cD5)
            cD4 = np.std(cD4)
            cD3 = np.std(cD3)
            cD2 = np.std(cD2)
            cD1 = np.std(cD1)
            temprow.append(cD1)
            temprow.append(cD2)
            temprow.append(cD3)
            temprow.append(cD4)
            temprow.append(cD5)

            if itr % 33 == 0:
                data.append(temprow.copy())
                temprow.clear()
            if itr == 1319:
                data.append(temprow.copy())
                temprow.clear()
            itr += 1
data = np.array(data)

out_data = pandas.DataFrame(data)
out_data.to_csv( 'WaveletTransformdp4.csv', mode='w', index=False)


import  csv
import numpy as np
import pandas
from matplotlib import mlab

mypath = "G:\\Datasets&GP\\DEAP\\data_preprocessed_python\\convertedData\\Frontal\\"


data = []
for i in range(1,33 ):
    name = ""
    if i < 10:
        name = mypath + "s0" + str(i)+"Frontal" + ".csv"
    else:
        name = mypath + "s" + str(i)+"Frontal" + ".csv"
    print(name)
    with open(name) as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        itr = 1
        temprow = []
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            temp = list(row.values())
            temp = [float(i) for i in temp]
            coeffs, freq = mlab.psd(temp, Fs=128)
            temprow.extend(list(coeffs))

            if itr % 12 == 0:
                data.append(temprow.copy())
                temprow.clear()
            itr += 1
data = np.array(data)

out_data = pandas.DataFrame(data)
out_data.to_csv( 'PowerSpectrumDenistyFrontal.csv', mode='w', index=False)

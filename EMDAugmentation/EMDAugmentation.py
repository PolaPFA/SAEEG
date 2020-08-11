import numpy as np
import pandas as pd
import csv
from PyEMD import EMD
#from emd import *

data_path = "E:\\College\\Graduation Project\\Dataset\\DEAP Dataset\\data_preprocessed_python\\data\\convertedData\\"
out_path = "E:\\College\\Graduation Project\\Dataset\\DEAP Dataset\\data_preprocessed_python\\data\\augmentedData\\"

name = data_path + 's01Frontal.csv'

emd = EMD()

session_num = 1
for i in range(1, 33):
    name = ""
    out_name = ''
    if i < 10:
        name = data_path + "s0" + str(i) + "Frontal.csv"
        #out_name = out_path + "s0" + str(i) + "Frontal.csv"
    else:
        name = data_path + "s" + str(i) + "Frontal.csv"
        #out_name = out_path + "s" + str(i) + "Frontal.csv"
    print(name)



    with open(name) as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        data = []
        itr = 1
        electrode_num = 1
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            temp = list(row.values())
            temp = [float(i) for i in temp]
            temp = np.array(temp)
            data.append(temp.copy())
            IMFS = emd(temp)
            for k in IMFS:
                data.append(k)
            out_name = out_path + "sess" + str(session_num).zfill(4) + "_electrode" + str(electrode_num).zfill(2) + ".csv"
            print('Session ' + str(session_num) + 'Electrode ' + str(electrode_num))
            data = pd.DataFrame(data)
            data.to_csv(out_name, index=False)
            data = []
            electrode_num += 1
            if itr % 12 == 0:
                session_num += 1
                electrode_num = 1
            itr += 1

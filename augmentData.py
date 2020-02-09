import  csv
from collections import defaultdict
import numpy as np
import pandas

mypath = "E:/4 Year/GP/Dataset/DEAP Dataset/data_preprocessed_python/data_preprocessed_python/"
columns = defaultdict(list)

for i in range(32):
    faster = []
    slower = []
    if i < 10:
        name = '%0*d' % (2,i+1)
    else:
        name = i+1
    fname = mypath+'convertedData/s'+str(name)+".csv"
    with open(fname) as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            temp =list(row.values())
            temp = [float(i) for i in temp]
            n = len(temp)
            temp2 = np.interp(np.linspace(0, n, 1.05 * n + 1), np.arange(n), temp)
            temp3 = np.interp(np.linspace(0, n, 0.95 * n + 1), np.arange(n), temp)
            slower.append(temp2)
            faster.append(temp3)
        fast = pandas.DataFrame(faster)
        slow = pandas.DataFrame(slower)
        fast.to_csv(mypath + 'interpolatesData\\f' + str(name) + '.csv', mode='w', index=False)
        slow.to_csv(mypath + 'interpolatesData\\s' + str(name) + '.csv', mode='w', index=False)
        print(i)


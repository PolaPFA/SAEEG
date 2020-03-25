import csv
import numpy as np
import pywt
import pandas
fout_data = []
mypath = "F:\\Datasets&GP\\DEAP\\data_preprocessed_python\\" #To be handled

for i in range(32):
    if i < 10:
        name = '%0*d' % (2,i+1)
    else:
        name = i+1
    fname = mypath+'convertedData\\s'+str(name)+".csv"
    print(i)
    data_row = []

    with open(fname) as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        c = 1
        for row in reader:
            data = row.values()
            data = [float(i) for i in data]
            data = data[0:7680]
            data_mean = np.mean(data)
            data = [i-data_mean for i in data]
            minimum = min(data)
            range = max(data) - minimum
            data = [float(i - minimum)/range for i in data]
            i = 0
            if c % 33 == 0 :
                fout_data.append(data_row)
                data_row = []
                c += 1
                continue
            while i < len(data):
                sample_rate = 128
                scale = 4
                temp = data[i:i+(scale*sample_rate)]
                coeffs = pywt.wavedec(temp, 'db4', level=6)
                energy = [np.sum(np.square(val)) for val in [coeff for coeff in coeffs]]
                entropy = [np.sum(np.square(val)*np.log(np.square(val))) for val in [coeff for coeff in coeffs]]
                for k in energy:
                    data_row.append(k)
                for k in entropy:
                    data_row.append(k)
                i = i + (2 * sample_rate)
            if c == 1319 :
                fout_data.append(data_row)
                data_row = []
            c +=1
   # data_row = pandas.Series(np.asarray(data_row).flatten())
fout_data = pandas.DataFrame(fout_data)
fout_data.to_csv('trainARM2.csv',mode= 'w',index=False)

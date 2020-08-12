import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

FeaturePath = "D:/1- FCIS ASU/4th Year/GP/DEAP Dataset/data_preprocessed_python/Old Data/"
LabelsPath = "D:/1- FCIS ASU/4th Year/GP/DEAP Dataset/data_preprocessed_python/Old Data/"
X = []
Y = []
Xno = []
Yno = []
DB4v = 0
DB10v = 0
PSDv = 0
DB4 = 0
DB10 = 0
PSD = 0
with open(LabelsPath+"convertedData/label0.csv") as f:
    reader = csv.DictReader(f)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        temp = list(row.values())
        temp = [float(i) for i in temp]
        temp2 = temp[0]
        if temp2 > 4.5:
            Y.append(1)
        else:
            Y.append(0)
with open(LabelsPath+"convertedData/label1.csv") as f:
    reader = csv.DictReader(f)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        temp = list(row.values())
        temp = [float(i) for i in temp]
        temp2 = temp[0]
        if temp2 > 4.5:
            Yno.append(1)
        else:
            Yno.append(0)

with open(FeaturePath+"featureswithout.csv") as f:
    reader = csv.DictReader(f)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        temp = list(row.values())
        #temp = [map(float,i) for i in temp]
        temp = [float(i) for i in temp]
        Xno.append(temp)
for i in range (32):
    X_train, X_test, Y_train, Y_test = train_test_split(Xno[i*40:(i+1)*40], Y[i*40:(i+1)*40], test_size=0.2,random_state=30)
    Model = RandomForestClassifier(max_depth=100, random_state=0)
    Model.fit(X_train, Y_train)
    scr = Model.score(X_test, Y_test)
    print('Velancy Accuracy Random Forest On Wavelet Transform Using db4: ', scr)
    DB4v += scr
    X_train, X_test, Y_train, Y_test = train_test_split(Xno[i * 40:(i + 1) * 40], Yno[i * 40:(i + 1) * 40],
                                                        test_size=0.2, random_state=30)
    Model = RandomForestClassifier(max_depth=100, random_state=0)
    Model.fit(X_train, Y_train)
    scr = Model.score(X_test, Y_test)
    print('Arosal Accuracy Random Forest On Wavelet Transform Using db4: ', scr)
    DB4 += scr
Xno.clear()

with open(FeaturePath+"featuresHorizontal-db10.csv") as f:
    reader = csv.DictReader(f)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        temp = list(row.values())
        #temp = [map(float,i) for i in temp]
        temp = [float(i) for i in temp]
        Xno.append(temp)


for i in range(32):
    X_train, X_test, Y_train, Y_test = train_test_split(Xno[i*40:(i+1)*40], Y[i*40:(i+1)*40], test_size=0.2,random_state=30)
    Model = RandomForestClassifier(max_depth=100, random_state=0)
    Model.fit(X_train, Y_train)
    scr = Model.score(X_test, Y_test)
    print('Velancy Accuracy Random Forest On Wavelet Transform Using db10 : ', scr)
    DB10v += scr
    X_train, X_test, Y_train, Y_test = train_test_split(Xno[i * 40:(i + 1) * 40], Yno[i * 40:(i + 1) * 40],
                                                        test_size=0.2, random_state=30)
    Model = RandomForestClassifier(max_depth=100, random_state=0)
    Model.fit(X_train, Y_train)
    scr = Model.score(X_test, Y_test)
    print('Arosal Accuracy Random Forest On Wavelet Transform Using db10 : ', scr)
    DB10 += scr
Xno.clear()

with open(FeaturePath+"featuresPowerSpectrumwithDWT.csv") as f:
    reader = csv.DictReader(f)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        temp = list(row.values())
        #temp = [map(float,i) for i in temp]
        temp = [float(i) for i in temp]
        Xno.append(temp)

for i in range(32):
    X_train, X_test, Y_train, Y_test = train_test_split(Xno[i*40:(i+1)*40], Y[i*40:(i+1)*40], test_size=0.2, random_state=30)
    Model = RandomForestClassifier(max_depth=50, random_state=0)
    Model.fit(X_train, Y_train)
    scr = Model.score(X_test, Y_test)
    print('Velancy Accuracy Random Forest On Wavelet Transform Using db4 and PSD : ', scr)
    PSDv += scr
    X_train, X_test, Y_train, Y_test = train_test_split(Xno[i * 40:(i + 1) * 40], Yno[i * 40:(i + 1) * 40],
                                                        test_size=0.2, random_state=30)
    Model = RandomForestClassifier(max_depth=50, random_state=0)
    Model.fit(X_train, Y_train)
    scr = Model.score(X_test, Y_test)
    print('Arosal Accuracy Random Forest On Wavelet Transform Using db4 and PSD : ', scr)
    PSD += scr
Xno.clear()
print('Average Velancy Accuracy Random Forest On Wavelet Transform Using db4: ', DB4v / 32)
print('Average Velancy Accuracy Random Forest On Wavelet Transform Using db10 : ', DB10v / 32)
print('Average Velancy Accuracy Random Forest On Wavelet Transform Using db4 and PSD : ', PSDv / 32)
print('Average Arosal Accuracy Random Forest On Wavelet Transform Using db4: ', DB4 / 32)
print('Average Arosal Accuracy Random Forest On Wavelet Transform Using db10 : ', DB10 / 32)
print('Average Arosal Accuracy Random Forest On Wavelet Transform Using db4 and PSD : ', PSD / 32)

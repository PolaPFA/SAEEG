import csv
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

FeaturePath = "D:/1- FCIS ASU/4th Year/GP/DEAP Dataset/data_preprocessed_python/Old Data/"
LabelsPath = "D:/1- FCIS ASU/4th Year/GP/DEAP Dataset/data_preprocessed_python/Old Data/"
X = []
Y = []
Xno = []
Yno = []

adaboostDB4v = 0
adaboostDB10v= 0
adaboostPSv = 0
adaboostDB4 = 0
adaboostDB10 = 0
adaboostPS = 0
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

for i in range(32):
    X_train, X_test, Y_train, Y_test = train_test_split(Xno[i*40:(i+1)*40], Y[i*40:(i+1)*40], test_size=0.2, random_state=5)
    Model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=30), n_estimators=60)
    Model.fit(X_train, Y_train)
    scr = Model.score(X_test, Y_test)
    adaboostDB4v += scr
    print('Velancy Accuracy Adaboost On Wavelet Transform Using db4: ', scr)
    X_train, X_test, Y_train, Y_test = train_test_split(Xno[i*40:(i+1)*40], Y[i*40:(i+1)*40], test_size=0.2,random_state=5)
    Model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=30), n_estimators=60)
    Model.fit(X_train, Y_train)
    scr = Model.score(X_test, Y_test)
    adaboostDB4 += scr
    print('Arosal Accuracy Adaboost On Wavelet Transform Using db4: ', scr)
Xno.clear()

with open(FeaturePath+"featuresHorizontal-db10.csv") as f:
    reader = csv.DictReader(f)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        temp = list(row.values())
        #temp = [map(float,i) for i in temp]
        temp = [float(i) for i in temp]
        Xno.append(temp)
for i in range(32):
    X_train, X_test, Y_train, Y_test = train_test_split(Xno[i*40:(i+1)*40], Y[i*40:(i+1)*40], test_size=0.2,random_state=5)
    Model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=150), n_estimators=150)
    Model.fit(X_train, Y_train)
    scr = Model.score(X_test, Y_test)
    print('Velancy Accuracy Adaboost On Wavelet Transform Using db10 : ', scr)
    adaboostDB10v += scr
    X_train, X_test, Y_train, Y_test = train_test_split(Xno[i*40:(i+1)*40], Y[i*40:(i+1)*40], test_size=0.2, random_state=5)
    Model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=150), n_estimators=150)
    Model.fit(X_train, Y_train)
    scr = Model.score(X_test, Y_test)
    print('Arosal Accuracy Adaboost On Wavelet Transform Using db10 : ', scr)
    adaboostDB10 += scr
Xno.clear()

with open(FeaturePath+"featuresPowerSpectrumwithDWT.csv") as f:
    reader = csv.DictReader(f)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        temp = list(row.values())
        #temp = [map(float,i) for i in temp]
        temp = [float(i) for i in temp]
        Xno.append(temp)
for i in range(32):
    X_train, X_test, Y_train, Y_test = train_test_split(Xno[i*40:(i+1)*40], Y[i*40:(i+1)*40], test_size=0.2,random_state=5)
    Model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=100), n_estimators=100)
    Model.fit(X_train, Y_train)
    scr = Model.score(X_test, Y_test)
    adaboostPSv += scr
    print('Velancy Accuracy Adaboost On Wavelet Transform Using db4 and PSD : ', scr)
    X_train, X_test, Y_train, Y_test = train_test_split(Xno[i*40:(i+1)*40], Y[i*40:(i+1)*40], test_size=0.2, random_state=5)
    Model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=100), n_estimators=100)
    Model.fit(X_train, Y_train)
    scr = Model.score(X_test, Y_test)
    print('Arosal Accuracy Adaboost On Wavelet Transform Using db4 and PSD : ', scr)
    adaboostPS += scr
Xno.clear()
print('Average Velancy Accuracy Adaboost On Wavelet Transform Using db4: ', adaboostDB4v / 32)
print('Average Velancy Accuracy Adaboost On Wavelet Transform Using db10 : ', adaboostDB10v / 32)
print('Average Velancy Accuracy Adaboost On Wavelet Transform Using db4 and PSD : ', adaboostPSv / 32)
print('Average Arosal Accuracy Adaboost On Wavelet Transform Using db4: ', adaboostDB4 / 32)
print('Average Arosal Accuracy Adaboost On Wavelet Transform Using db10 : ', adaboostDB10 / 32)
print('Average Arosal Accuracy Adaboost On Wavelet Transform Using db4 and PSD : ', adaboostPS / 32)


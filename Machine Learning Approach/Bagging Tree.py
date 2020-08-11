import csv
from sklearn.ensemble import BaggingClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split

FeaturePath = "D:/1- FCIS ASU/4th Year/GP/DEAP Dataset/data_preprocessed_python/Old Data/"
LabelsPath = "D:/1- FCIS ASU/4th Year/GP/DEAP Dataset/data_preprocessed_python/Old Data/"
X = []
Y = []
Xno = []
Yno = []

c = 1
arr = []
with open(LabelsPath+"convertedData/label0.csv") as f:
    reader = csv.DictReader(f)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        temp = list(row.values())
        temp = [float(i) for i in temp]
        temp2 = temp[0]
        if temp2 > 4.5:
            Y.append(1)
            arr.append(1)
        else:
            Y.append(0)
            arr.append(0)
        if c % 40 == 0:
            for item in arr:
                Y.append(item)
            arr = []
        c += 1
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
#Replace featureswithout
with open(FeaturePath+"featureswithout.csv") as f:
    reader = csv.DictReader(f)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        temp = list(row.values())
        #temp = [map(float,i) for i in temp]
        temp = [float(i) for i in temp]
        Xno.append(temp)
X_train, X_test, Y_train, Y_test = train_test_split(Xno, Yno, test_size=0.2,random_state=5)
Model = BaggingClassifier(base_estimator=svm.SVC(), n_estimators=15).fit(X_train, Y_train)
Model.fit(X_train, Y_train)
scr = Model.score(X_test, Y_test)
print('Accuracy Bagging Tree On Wavelet Transform Using db4: ', scr)
predict_val = Model.predict(X_test)
Xno.clear()

with open(FeaturePath+"featuresHorizontal-db10.csv") as f:
    reader = csv.DictReader(f)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        temp = list(row.values())
        #temp = [map(float,i) for i in temp]
        temp = [float(i) for i in temp]
        Xno.append(temp)
X_train, X_test, Y_train, Y_test = train_test_split(Xno, Yno, test_size=0.2,random_state=5)
Model = BaggingClassifier(base_estimator=svm.SVC(kernel='rbf', C=0.1, gamma='scale', degree=5), n_estimators=15).fit(X_train, Y_train)
Model.fit(X_train, Y_train)
scr = Model.score(X_test, Y_test)
print('Accuracy Bagging Tree On Wavelet Transform Using db10 : ', scr)
predict_val = Model.predict(X_test)
Xno.clear()

with open(FeaturePath+"featuresPowerSpectrumwithDWT.csv") as f:
    reader = csv.DictReader(f)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        temp = list(row.values())
        #temp = [map(float,i) for i in temp]
        temp = [float(i) for i in temp]
        Xno.append(temp)
X_train, X_test, Y_train, Y_test = train_test_split(Xno, Yno, test_size=0.2, random_state=5)
Model = BaggingClassifier(base_estimator=svm.SVC(kernel='rbf', C=1, gamma='scale', degree=5), n_estimators=15).fit(X_train, Y_train)
Model.fit(X_train, Y_train)
scr = Model.score(X_test, Y_test)
print('Accuracy Bagging Tree On Wavelet Transform Using db4 and PSD : ', scr)
predict_val = Model.predict(X_test)
Xno.clear()

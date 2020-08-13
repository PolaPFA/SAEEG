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

DB4V = 0
DB10V = 0
PSV = 0
DB4 = 0
DB10 = 0
PS = 0
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
#Replace featureswithout
with open(FeaturePath+"featureswithout.csv") as f:
    reader = csv.DictReader(f)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        temp = list(row.values())
        #temp = [map(float,i) for i in temp]
        temp = [float(i) for i in temp]
        Xno.append(temp)
for i in range(32):
    X_train, X_test, Y_train, Y_test = train_test_split(Xno[i * 40:(i + 1) * 40], Y[i * 40:(i + 1) * 40],
                                                        test_size=0.2, random_state=5)
    Model = BaggingClassifier(base_estimator=svm.SVC(), n_estimators=15).fit(X_train, Y_train)
    scr = Model.score(X_test, Y_test)
    DB4V += scr
    print('Velancy Accuracy Bagging Tree On Wavelet Transform Using db4: ', scr)
    X_train, X_test, Y_train, Y_test = train_test_split(Xno[i*40:(i+1)*40], Yno[i*40:(i+1)*40], test_size=0.2,random_state=5)
    Model = BaggingClassifier(base_estimator=svm.SVC(), n_estimators=15).fit(X_train, Y_train)
    scr = Model.score(X_test, Y_test)
    DB4 += scr
    print('Arosal Accuracy Bagging Tree On Wavelet Transform Using db4: ', scr)
Xno.clear()

with open(FeaturePath+"featuresHorizontal-db10.csv") as f:
    reader = csv.DictReader(f)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        temp = list(row.values())
        #temp = [map(float,i) for i in temp]
        temp = [float(i) for i in temp]
        Xno.append(temp)
for i in range(32):
    X_train, X_test, Y_train, Y_test = train_test_split(Xno[i * 40:(i + 1) * 40], Y[i * 40:(i + 1) * 40],
                                                        test_size=0.2, random_state=5)
    Model = BaggingClassifier(base_estimator=svm.SVC(kernel='rbf', C=0.1, gamma='scale', degree=5),
                              n_estimators=15).fit(X_train, Y_train)
    scr = Model.score(X_test, Y_test)
    DB10V += scr
    print('Velancy Accuracy Bagging Tree On Wavelet Transform Using db10 : ', scr)
    X_train, X_test, Y_train, Y_test = train_test_split(Xno[i*40:(i+1)*40], Yno[i*40:(i+1)*40], test_size=0.2,random_state=5)
    Model = BaggingClassifier(base_estimator=svm.SVC(kernel='rbf', C=0.1, gamma='scale', degree=5), n_estimators=15).fit(X_train, Y_train)
    scr = Model.score(X_test, Y_test)
    DB10 += scr
    print('Arosal Accuracy Bagging Tree On Wavelet Transform Using db10 : ', scr)
Xno.clear()


with open(FeaturePath+"featuresPowerSpectrumwithDWT.csv") as f:
    reader = csv.DictReader(f)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        temp = list(row.values())
        #temp = [map(float,i) for i in temp]
        temp = [float(i) for i in temp]
        Xno.append(temp)
for i in range(32):
    X_train, X_test, Y_train, Y_test = train_test_split(Xno[i * 40:(i + 1) * 40], Y[i * 40:(i + 1) * 40],
                                                        test_size=0.2, random_state=5)
    Model = BaggingClassifier(base_estimator=svm.SVC(kernel='rbf', C=1, gamma='scale', degree=5), n_estimators=15).fit(
        X_train, Y_train)
    scr = Model.score(X_test, Y_test)
    PSV += scr
    print('Velancy Accuracy Bagging Tree On Wavelet Transform Using db4 and PSD : ', scr)
    X_train, X_test, Y_train, Y_test = train_test_split(Xno[i*40:(i+1)*40], Yno[i*40:(i+1)*40], test_size=0.2,random_state=5)
    Model = BaggingClassifier(base_estimator=svm.SVC(kernel='rbf', C=1, gamma='scale', degree=5), n_estimators=15).fit(X_train, Y_train)
    scr = Model.score(X_test, Y_test)
    PS += scr
    print('Arosal Accuracy Bagging Tree On Wavelet Transform Using db4 and PSD : ', scr)
Xno.clear()

print('Average Velancy Accuracy Bagging Tree On Wavelet Transform Using db4: ', DB4V/32)
#print('Average Arosal Accuracy Bagging Tree On Wavelet Transform Using db4: ', DB4/32)
print('Average Velancy Accuracy Bagging Tree On Wavelet Transform Using db10 : ', DB10V/32)
#print('Average Arosal Accuracy Bagging Tree On Wavelet Transform Using db10 : ', DB10/32)
print('Average Velancy Accuracy Bagging Tree On Wavelet Transform Using db4 and PSD : ', PSV/32)
#print('Average Arosal Accuracy Bagging Tree On Wavelet Transform Using db4 and PSD : ', PS/32)

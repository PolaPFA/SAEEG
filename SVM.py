import  csv
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas

mypath = "E:/4 Year/GP/Dataset/DEAP Dataset/data_preprocessed_python/data_preprocessed_python/"
X = []
Y = []
with open(mypath+"features.csv") as f:
    reader = csv.DictReader(f)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        temp = list(row.values())
        temp = [float(i) for i in temp]
        X.append(temp)
for i in range(3):
    with open(mypath+"convertedData\\labels_0.csv") as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            temp = list(row.values())
            temp = [float(i) for i in temp]
            temp2 = temp[0]
            if temp2 > 4.5:
                Y.append(1)
            else:
                Y.append(0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
clf = svm.SVC(kernel='rbf', C=1, gamma='scale', degree=5)
clf.fit(X_train, Y_train)
scr = clf.score(X_test, Y_test)
print('Accuracy : ', scr)
predict_val = clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test,predict_val))
print(classification_report(Y_test,predict_val))



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
c = 1
arr = []
with open(mypath+"convertedData\\labels_0.csv") as f:
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
                Y.append(item)
            arr = []
        c += 1
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#SVM
clf = svm.SVC(kernel='rbf', C=1, gamma='scale', degree=5)
clf.fit(X_train, Y_train)
scr = clf.score(X_test, Y_test)
print('Accuracy Svm: ', scr)
predict_val = clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
#print(confusion_matrix(Y_test,predict_val))
#print(classification_report(Y_test,predict_val))

#Knn
import sklearn.neighbors as nb
knn = nb.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
scr = knn.score(X_test, Y_test)
print('Accuracy Knn: ', scr)

#Tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=200)
ada.fit(X_train, Y_train)
scr = ada.score(X_test, Y_test)
print('Accuracy adaboost: ', scr)

#Logistic
from sklearn import linear_model
logReg1 = linear_model.LogisticRegression(penalty='l2', C=1)
logReg1.fit(X_train, Y_train)
scr = logReg1.score(X_train, Y_train)
print('Accuracy logistic: ', scr)

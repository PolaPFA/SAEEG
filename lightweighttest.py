from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from joblib import dump, load
from tensorflow import keras
import joblib
import csv

import os
test=[]

for i in range(10):
    string = "fold0_"
    test.append(joblib.load("ModelsandData\\"+string+"test_"+str(i+1)+" .joblib"))
flat_list = []
for sublist in test:
    for item in sublist:
        flat_list.append(item)
res = []
[res.append(x) for x in flat_list if x not in res]

X = []
Y0= []
Y1=[]
for i in range(1):
    print("label0.csv")
    with open("label0.csv") as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            temp = list(row.values())
            temp = [float(i) for i in temp]
            temp2 = temp[0]
            if temp2 >= 5:
                Y0.append(1)
            else:
                Y0.append(0)
    print("label1.csv")
    with open("label1.csv") as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            temp = list(row.values())
            temp = [float(i) for i in temp]
            temp2 = temp[0]
            if temp2 >= 5:
                Y1.append(1)
            else:
                Y1.append(0)


for i in range(1,33):
    name = ""
    if i < 10:
        name =  "E:\\College\\Graduation Project\\Dataset\\DEAP Dataset\\data_preprocessed_python\\data\\convertedData\\s0" + str(i)+"Frontal"
    else:
        name = "E:\\College\\Graduation Project\\Dataset\\DEAP Dataset\\data_preprocessed_python\\data\\convertedData\\s" + str(i)+"Frontal"
    print(name)
    with open(name+".csv") as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        itr=1
        temprow = []
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            temp = list(row.values())
            temp = [float(i) for i in temp]
            temprow.append(temp.copy())
            if itr%12 == 0:
                X.append(temprow.copy())
                temprow.clear()
            itr+=1
def maxvote(list):
    zerocount =0
    onecount =0
    for i in list:
        if i == 0:
            zerocount+=1
        else:
            onecount+=1
    if onecount >zerocount:
        return  onecount
    else:
        return  zerocount

rightanswer=0

for item in res:
    reading=[]
    for i in range(12):
        electrode=[]
        for j in range(10):
            string="folds_0 "
            Model= keras.models.load_model("ModelsandData\\"+string+"electrode"+str(i+1)+"fold"+str(j+1)+".joblib")
            temp= Model.predict(X[item])
            electrode.append(temp)
        val =maxvote(electrode)
        reading.append(val)
    finalval =maxvote(reading)
    if(Y0[item] == finalval):
        rightanswer+=1

print("acc = " + str((rightanswer / len(res))) * 100)

        #then append the result of max vote in reading
    #check if right or wrong

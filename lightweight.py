import tensorflow as tf
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from joblib import dump, load

import os



# define the name of the directory to be created


#mypath="G:\\Datasets&GP\\DEAP\\data_preprocessed_python\\convertedData\\"
mypath = "E:\\College\\Graduation Project\\Dataset\\DEAP Dataset\\data_preprocessed_python\\data\\convertedData\\"

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
        name = mypath + "s0" + str(i)+"Frontal"
    else:
        name = mypath + "s" + str(i)+"Frontal"
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
print(X.shape)
model = tf.keras.models.Sequential()

l1=tf.keras.layers.Conv1D( filters=32, kernel_size=5,input_shape=(8064,1),strides=3)
model.add(l1)
print(l1.output_shape)
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
model.add(tf.keras.layers.Conv1D(filters=24, kernel_size=3,strides=2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
model.add(tf.keras.layers.Conv1D(filters=16,kernel_size= 3,strides=2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
lcov=tf.keras.layers.Conv1D(filters=8,kernel_size= 5 ,strides=3)
model.add(lcov)
print(lcov.output_shape)
lbatch=tf.keras.layers.BatchNormalization()
model.add(lbatch)
print(lbatch.output_shape)
lact=tf.keras.layers.Activation(tf.keras.activations.relu)
model.add(lact)
print(lact.output_shape)
#lflat=tf.keras.layers.Reshape((8),input_shape=(None,None,8))
#tf.keras.layers.Flatten(input_shape=(None,None,8))
#model.add(lflat)
#print(lflat.output)
model.add(tf.keras.layers.Flatten())
ldense=tf.keras.layers.Dense(20, input_shape=(None,None,20),activation='relu')
model.add(ldense)
print(ldense.output_shape)
model.add(tf.keras.layers.Dropout(.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
#model.add(tf.keras.layers.Softmax(axis=-1))

# model = tf.keras.models.Sequential([
#
#     tf.keras.layers.Conv1D(32, (1,5), activation='relu', input_shape=(None, 100),strides=3),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Conv1D(24, (1,3), activation='relu', strides=2),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Conv1D(16, (1,3), activation='relu', strides=2),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Conv1D(8, (1,5), activation='relu', strides=3),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dense(40, activation='relu'),
#     tf.keras.layers.Dropout(.5),
#     tf.keras.layers.Dense(2),
#     tf.keras.layers.Softmax(axis=-1)
#
# ])

folds_0 = []
folds_1 = []
modeel=1
for fo in range(10):
    models_0 = []
    models_1 = []
    for mo in range(12):
        print("model_number_intilize=" + str(modeel))

        current_model_0=tf.keras.models.clone_model(model)
        current_model_0.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                      optimizer=tf.keras.optimizers.Adam(lr=0.05),
                                      # tfkeras.optimizers.Adam(learning_rate=0.01)
                                      metrics=['acc'])

        models_0.append(current_model_0)
        print("model_number_intilize=" + str(modeel+1))
        current_model_1 = tf.keras.models.clone_model(model)
        current_model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                      optimizer=tf.keras.optimizers.Adam(lr=0.05),
                                      # tfkeras.optimizers.Adam(learning_rate=0.01)
                                      metrics=['acc'])
        modeel += 2
        models_1.append(current_model_1)
    folds_0.append(models_0)
    folds_1.append(models_1)


# X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y1, test_size = 0.2)
# X0_train, X0_test, Y0_train, Y0_test = train_test_split(X, Y0, test_size = 0.2)
#
# X1_train = np.transpose(X1_train, (1, 0, 2))
# X0_train = np.transpose(X0_train, (1, 0, 2))
# X1_test = np.transpose(X1_test, (1, 0, 2))
# X0_test = np.transpose(X0_test, (1, 0, 2))
#
# for i in range(12):
#     H0 = model.fit(X0_train[i], Y0_train[i], epochs=10)
#     channel_models_0.append(H0)
#
#     H1 = model.fit(X1_train[i], Y1_train[i], epochs=10)
#     channel_models_1.append(H1)
'''
X0_train_all, X0_test_all, Y0_train_all, Y0_test_all = train_test_split(X, Y0, shuffle=False, random_state=32,test_size=0.1)
X1_train_all, X1_test_all, Y1_train_all, Y1_test_all = train_test_split(X, Y1, shuffle=False, random_state=32,test_size=0.1)

dump(X0_test_all,"ModelsandData/X0_test_all .joblib")
dump(Y0_test_all,"ModelsandData/Y0_test_all .joblib")
dump(X1_test_all,"ModelsandData/X1_test_all .joblib")
dump(Y1_test_all,"ModelsandData/Y1_test_all .joblib")
'''

X0_train_all, X0_test_all, Y0_train_all, Y0_test_all = train_test_split(X, Y0, shuffle=False, random_state=32,test_size=0.2)
dump(X0_test_all,"ModelsandData/X0_test_all .joblib")
dump(Y0_test_all,"ModelsandData/Y0_test_all .joblib")


#Kfold = StratifiedKFold(n_splits=1 , shuffle=True)
scores_0 = []
itr = 0

#for train, test in Kfold.split(X, Y0):
for train, test in X0_train_all, Y0_train_all:
    dump(test, 'ModelsandData/fold0_test_'+str(itr+1)+" .joblib")
    X0_train =np.array(X)[train.astype(int)]
    X0_test = np.array(X)[test.astype(int)]
    Y0_train = np.array(Y0)[train.astype(int)]
    Y0_test = np.array(Y0)[test.astype(int)]
    X0_train = np.transpose(X0_train, (1, 0, 2))
    X0_test = np.transpose(X0_test, (1, 0, 2))

    for i in range(12):
        print("Valancy folds"+str(itr+1)+"electode : "+str(i+1))
        current_x = X0_train[i]
        current_x = np.expand_dims(current_x, axis = 1)
        current_x = np.transpose(current_x, (0, 2, 1))
        print(current_x.shape)
        current_y=np.expand_dims(Y0_train, axis = 0)
        current_y = np.transpose(current_y, (1, 0))
        print(current_y.shape)
        folds_0[itr][i].fit(current_x, current_y, epochs=50)
        current_x = X0_test[i]
        current_x = np.expand_dims(current_x, axis=1)
        current_x = np.transpose(current_x, (0, 2, 1))
        print(current_x.shape)
        current_y = np.expand_dims(Y0_test, axis=0)
        current_y = np.transpose(current_y, (1, 0))
        #model.evaluate(current_x, current_y)
        #dump(folds_0[itr][i], "ModelsandData/folds_0 electrode"+str(i+1)+"fold"+str(itr+1) +".joblib")
        path = "ModelsandData/folds_0 electrode"+str(i+1)+"fold"+str(itr+1)
        os.mkdir(path)
        folds_0[itr][i].save('ModelsandData/folds_0 electrode'+str(i+1)+"fold"+str(itr+1)+".joblib")

    itr+=1

'''
channel_models_1 = []
itr=0
for train, test in Kfold.split(X, Y1):
    dump(test, 'ModelsandData/fold1_test_' + str(itr+1)+" .joblib")

    X1_train = np.array(X)[train.astype(int)]
    X1_test = np.array(X)[test.astype(int)]
    Y1_train = np.array(Y1)[train.astype(int)]
    Y1_test = np.array(Y1)[test.astype(int)]
    X1_train = np.transpose(X1_train, (1, 0, 2))
    X1_test = np.transpose(X1_test, (1, 0, 2))

    for i in range(12):
        print("Arrosal folds : " + str(itr+1)+"electode : "+str(i+1))
        current_x = X1_train[i]
        current_x = np.expand_dims(current_x, axis=1)
        current_x = np.transpose(current_x, (0, 2, 1))
        print(current_x.shape)
        current_y = np.expand_dims(Y1_train, axis=0)
        current_y = np.transpose(current_y, (1, 0))
        print(current_y.shape)
        folds_1[itr,i].fit(current_x, current_y, epochs=50)
        current_x = X1_test[i]
        current_x = np.expand_dims(current_x, axis=1)
        current_x = np.transpose(current_x, (0, 2, 1))
        print(current_x.shape)
        current_y = np.expand_dims(Y1_test, axis=0)
        current_y = np.transpose(current_y, (1, 0))
        #model.evaluate(current_x, current_y)
        #dump(folds_1[itr,i], "ModelsandData/folds_1 electrode"+str(i+1)+"fold"+str(itr+1)+".joblib")
        path = "ModelsandData/folds_1 electrode" + str(i + 1) + "fold" + str(itr + 1)
        os.mkdir(path)
        #folds_1[itr][i].save('ModelsandData/folds_1 electrode' + str(i + 1) + "fold" + str(itr + 1))
        folds_1[itr][i].save_weights('ModelsandData/folds_1 electrode' + str(i + 1) + "fold" + str(itr + 1) + ".h5")

    itr+=1

'''
'''
    X1_train = np.transpose(X1_train, (1, 0, 2))

    X1_test = np.transpose(X1_test, (1, 0, 2))
    '''









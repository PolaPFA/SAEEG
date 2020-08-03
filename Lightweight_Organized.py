import tensorflow as tf
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from joblib import dump, load
import os


data_path = "G:\\Datasets&GP\\DEAP\\data_preprocessed_python\\convertedData\\"

def get_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=5, input_shape=(8064, 1), strides=3))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv1D(filters=24, kernel_size=3, strides=2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=3))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(20, input_shape=(None, None, 20), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    model.add(tf.keras.layers.Softmax())
    return model

def read_convert_output(file_name):
    data = []
    with open(file_name) as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            temp = list(row.values())
            temp = [float(i) for i in temp]
            temp2 = temp[0]
            if temp2 >= 5:
                data.append(1)
            else:
                data.append(0)
    data=np.array(data)
    return data


def read_input_data():

    data = []
    for i in range(1, 36):
        name = ""
        if i < 10:
            name = data_path + "s0" + str(i) + "Frontal.csv"
        else:
            name = data_path + "s" + str(i) + "Frontal.csv"
        print(name)
        with open(name) as f:
            reader = csv.DictReader(f)  # read rows into a dictionary format
            itr = 1
            temprow = []
            for row in reader:  # read a row as {column1: value1, column2: value2,...}
                temp = list(row.values())
                temp = [float(i) for i in temp]
                temprow.append(temp.copy())
                if itr % 12 == 0:
                    data.append(temprow.copy())
                    temprow.clear()
                itr += 1
    data=np.array(data)
    print(data.shape)
    return data

def convert_x_dimensions(input,itr):
    output = np.transpose(input, (1, 0, 2))
    current_x = output[itr]
    current_x = np.expand_dims(current_x, axis=1)
    current_x = np.transpose(current_x, (0, 2, 1))
    return current_x

def convert_y_dimensions(input):
    current_y = np.expand_dims(input, axis=0)
    return np.transpose(current_y)

def initialize_models(number_of_models):
    valence_models = []
    arousal_models = []

    for model in range(number_of_models):
        current_model_0 = tf.keras.models.clone_model(cnn_model)
        current_model_0.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                optimizer=tf.keras.optimizers.Adam(lr=0.05),
                                metrics=['acc'])
        valence_models.append(current_model_0)

        current_model_1 = tf.keras.models.clone_model(cnn_model)
        current_model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                optimizer=tf.keras.optimizers.Adam(lr=0.05),
                                metrics=['acc'])
        arousal_models.append(current_model_1)

    return valence_models, arousal_models

def maxvote(list):
    output = []
    for index in list:
        zerocount = len([list[index] == 0])
        onecount = len([list[index] == 1])
        if onecount > zerocount:
            output.append(1)
        else:
            output.append(0)
    return output

print('Before reading data')
X = read_input_data()
print(X.shape)
print('Before reading logits')
Y0 = read_convert_output('label0.csv')
Y1 = read_convert_output('label1.csv')
Y0=Y0[0:40]
Y1=Y1[0:40]
print('Before getting model')
cnn_model = get_model()

print('Before initializing models')
valence_models, arousal_models = initialize_models(12)

print('Before splitting')
X0_train, X0_test, Y0_train, Y0_test = train_test_split(X, Y0, shuffle=False, random_state=32, test_size=0.2)
print(X0_train.shape)
#dump(X0_test, "NewModelsandData/X0_test.joblib")
#dump(Y0_test, "NewModelsandData/Y0_test.joblib")

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y1, shuffle=False, random_state=32, test_size=0.2)
#dump(X1_test, "NewModelsandData/X1_test.joblib")
#dump(Y1_test, "NewModelsandData/Y1_test.joblib")

test_y1 = convert_y_dimensions(Y1_test)
test_y0 = convert_y_dimensions(Y0_test)

#Training the double models:
for i in range(12):
    print('Training Electrode number: ', i+1)
    print(X0_train.shape)
    train_x0 = convert_x_dimensions(X0_train,i)
    train_x1 = convert_x_dimensions(X1_train,i)
    train_y1 = convert_y_dimensions(Y1_train[i])
    train_y0 = convert_y_dimensions(Y0_train[i])
    print(train_x0.shape)
    print(Y0_train.shape)
    valence_models[i].fit(train_x0, Y0_train, epochs=100)
    arousal_models[i].fit(train_x1, Y1_train, epochs=100)

#Testing the double models:
valencies = []
arousals = []
for i in range(12):
    test_x0 = convert_x_dimensions(X0_test,i)
    test_x1 = convert_x_dimensions(X1_test,i)
    valencies.append(valence_models[i].predict(test_x0))
    arousals.append(arousal_models[i].predict(test_x1))
valencies = maxvote(valencies)
arousals = maxvote(arousals)

print('Valence accuracy:', accuracy_score(valencies, Y0_test))
print('Arousals accuracy:', accuracy_score(arousals, Y1_test))
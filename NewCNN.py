import tensorflow as tf
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from joblib import dump, load
import os

data_path = "E:\\College\\Graduation Project\\Dataset\\DEAP Dataset\\data_preprocessed_python\\data\\convertedData\\"
#data_path = "G:\\Datasets&GP\\DEAP\\data_preprocessed_python\\convertedData\\"

def get_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D())
    #model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3)))
    #model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    #model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3)))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.MaxPool2D())
    #model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5)))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    #model.add(tf.keras.layers.Softmax(axis=-1))
    return tf.keras.models.clone_model(model)

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
    for i in range(1, 33):
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
                    #data.append(convert_input_data(temprow))       #Converting it to 88*88 matrix instead of 1*8064 vector
                    temprow.clear()
                itr += 1
    data = np.array(data)
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

def convert_input_data(old_data):
    data = np.array(old_data)
    #Removing first 3 seconds (Not important)
    data = data[:, 3*128:]
    data = np.append(data, data[:, :64], axis=1)
    data = np.expand_dims(data, axis=2)
    data = data.reshape((12, 88, 88))
    return data


print('Before reading data')
X = read_input_data()
X = np.expand_dims(X, axis=3)
#X = np.transpose(X, (0, 2, 3, 1))
print(X.shape)

input_shape = X.shape[1:]


print('Before reading logits')
Y0 = read_convert_output('label0.csv')
Y1 = read_convert_output('label1.csv')
#Y0 = Y0[:400]
#Y1 = Y1[:400]
print('Before getting model')
cnn_valence_model = get_model()
cnn_valence_model.build((None,input_shape))
cnn_valence_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                optimizer=tf.keras.optimizers.Adam(),
                                metrics=['acc'])

cnn_arousal_model = get_model()
cnn_arousal_model.build((None, input_shape))
cnn_arousal_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                optimizer=tf.keras.optimizers.Adam(),
                                metrics=['acc'])


print('Before splitting')
X0_train, X0_test, Y0_train, Y0_test = train_test_split(X, Y0, shuffle=True, random_state=32, test_size=0.2)


X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y1, shuffle=True, random_state=32, test_size=0.2)

train_y0 = convert_y_dimensions(Y0_train)
train_y1 = convert_y_dimensions(Y0_train)
test_y1 = convert_y_dimensions(Y1_test)
test_y0 = convert_y_dimensions(Y0_test)

print(X0_train.shape, X0_test.shape, train_y0.shape, test_y0.shape)
print(X1_train.shape, X1_test.shape, train_y1.shape, test_y1.shape)

#Training
cnn_valence_model.fit(X0_train, train_y0, epochs=300)
cnn_arousal_model.fit(X1_train, train_y1, epochs=300)

#Testing the double models:
print('Valence models:')
print('Train: ', cnn_valence_model.evaluate(X0_train, train_y0))
print('Test: ', cnn_valence_model.evaluate(X0_test, test_y0))

print('Arousal models:')
print('Train: ', cnn_arousal_model.evaluate(X1_train, train_y1))
print('Test: ', cnn_arousal_model.evaluate(X1_test, test_y1))
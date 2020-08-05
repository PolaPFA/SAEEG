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

def get_model():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=3,input_shape=(59, 1), strides=2))

    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dense(20, activation='relu'))

    model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=2))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dense(20, activation='relu'))

    model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=2))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dense(20, activation='relu'))

    model.add(tf.keras.layers.Conv1D(filters=8, kernel_size=5, strides=3))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(30, input_shape=(None, None, 30), activation='relu'))
    model.add(tf.keras.layers.Dense(30, activation='relu'))
    model.add(tf.keras.layers.Dense(30, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
   # model.add(tf.keras.layers.Dense(2, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    #model.add(tf.keras.layers.Softmax())
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
    return data


def read_input_data(name):
    print(name)
    data=[]
    with open(name) as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            temp = list(row.values())
            temp = [float(i) for i in temp]
            data.append(temp.copy())
    return data

def convert_x_dimensions(input):
    output = np.expand_dims(input, axis=1)
    return np.transpose(output, (0, 2, 1))

def convert_y_dimensions(input):
    current_y = np.expand_dims(input, axis=0)
    return np.transpose(current_y)

def initialize_models(number_of_models):
    valence_models = []
    arousal_models = []

    for model in range(number_of_models):
        current_model_0 = tf.keras.models.clone_model(cnn_model)
        current_model_0.compile(loss=tf.keras.losses.mean_squared_error,
                                optimizer=tf.keras.optimizers.RMSprop(lr=0.125),
                                metrics=['acc'])
        valence_models.append(current_model_0)

        current_model_1 = tf.keras.models.clone_model(cnn_model)
        current_model_1.compile(loss=tf.keras.losses.mean_squared_error,
                                optimizer=tf.keras.optimizers.RMSprop(lr=0.125),
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
X = read_input_data("WaveletTransform&PowerSpectrumFrontal .csv")

print('Before reading logits')
Y0 = read_convert_output('label0.csv')
Y1 = read_convert_output('label1.csv')

print('Before getting model')
cnn_model = get_model()

print('Before initializing models')
valence_models, arousal_models = initialize_models(1)

print('Before splitting')
X0_train, X0_test, Y0_train, Y0_test = train_test_split(X, Y0, shuffle=True, random_state=32, test_size=0.2)
#dump(X0_test, "NewModelsandData/X0_test.joblib")
#dump(Y0_test, "NewModelsandData/Y0_test.joblib")

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y1, shuffle=True, random_state=32, test_size=0.2)
#dump(X1_test, "NewModelsandData/X1_test.joblib")
#dump(Y1_test, "NewModelsandData/Y1_test.joblib")



#Training the double models:


print('Training Valancy : ')
X0_train=convert_x_dimensions(X0_train)
Y0_train=convert_y_dimensions(Y0_train)
valence_models[0].fit(X0_train, Y0_train, epochs=2000)
print('Training Arosal : ')
X1_train=convert_x_dimensions(X1_train)
Y1_train=convert_y_dimensions(Y1_train)
arousal_models[0].fit(X1_train, Y1_train, epochs=2000)

#Testing the double models:


print('Testing Valancy : ')
X0_test=convert_x_dimensions(X0_test)
Y0_test=convert_y_dimensions(Y0_test)
valence_models[0].evaluate(X0_test, Y0_test)
print('Testing Arosal : ')
X1_test=convert_x_dimensions(X1_test)
Y1_test=convert_y_dimensions(Y1_test)
arousal_models[0].fit(X1_test, Y1_test)
'''
Model 

16 filter kernel 3 stride 2
batch Normalization
activation relu

8  filter kernel 5 stride 3
batch Normalization
activation relu
flatten layer
Dense 30 activation relu
Dense 30 activation relu
Dense 30 activation relu
 Drop out 0.5
Dense 20 activation relu
Dense 20 activation relu
 Drop out 0.5
 Dense 10 activation relu
Dense 10 activation relu
 Drop out 0.5
Dense 1 sigmoid

epoch=2000
Learning rate =0.125
Valancy 58.98%
arosal 60.94%
______________________________________________________________________________________________________________________

Model 

16 filter kernel 3 stride 2
batch Normalization
activation relu
16 filter kernel 3 stride 2
batch Normalization
activation relu
16 filter kernel 3 stride 2
batch Normalization
activation relu
8  filter kernel 5 stride 3
batch Normalization
activation relu
flatten layer
Dense 30 activation relu
Dense 30 activation relu
Dense 30 activation relu
 Drop out 0.5
Dense 20 activation relu
Dense 20 activation relu
 Drop out 0.5
 Dense 10 activation relu
Dense 10 activation relu
 Drop out 0.5
Dense 1 sigmoid

epoch=2000
Learning rate =0.125
Valancy 58.98%
arosal 54.30%

_______________________________________________________________________________________________________________________

16 filter kernel 3 stride 2
batch Normalization
activation relu
Dense 20 activation relu
Dense 20 activation relu
 Drop out 0.5
16 filter kernel 3 stride 2
batch Normalization
activation relu
16 filter kernel 3 stride 2
batch Normalization
activation relu
8  filter kernel 5 stride 3
batch Normalization
activation relu
flatten layer
Dense 30 activation relu
Dense 30 activation relu
Dense 30 activation relu
 Drop out 0.5
Dense 20 activation relu
Dense 20 activation relu
 Drop out 0.5
 Dense 10 activation relu
Dense 10 activation relu
 Drop out 0.5
Dense 1 sigmoid
RMS PROP Optimizer

epoch=2000
Learning rate =0.125
Valancy 58.98%
arosal 60.55%

______________________________________________________________________________________________________________________

16 filter kernel 3 stride 2
batch Normalization
activation relu
Dense 20 activation relu
Dense 20 activation relu

16 filter kernel 3 stride 2
batch Normalization
activation relu
Dense 20 activation relu
Dense 20 activation relu

16 filter kernel 3 stride 2
batch Normalization
activation relu
Dense 20 activation relu
Dense 20 activation relu

8  filter kernel 5 stride 3
batch Normalization
activation relu
flatten layer
Dense 30 activation relu
Dense 30 activation relu
Dense 30 activation relu
 Drop out 0.5
Dense 20 activation relu
Dense 20 activation relu
 Drop out 0.5
 Dense 10 activation relu
Dense 10 activation relu
 Drop out 0.5
Dense 1 sigmoid
RMS PROP Optimizer

epoch=2000
Learning rate =0.125
Valancy 58.98%
arosal 60.94%
_______________________________________________________________________________________________________________________
'''
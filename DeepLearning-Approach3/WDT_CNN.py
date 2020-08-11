import tensorflow as tf
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split


def get_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, input_shape=input_shape,
                                     kernel_initializer=tf.keras.initializers.RandomNormal()))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=3))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.9))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
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

def convert_y_dimensions(input):
    current_y = np.expand_dims(input, axis=0)
    return np.transpose(current_y)

input_shape = (165, 1)

dataframe = pd.read_csv('WaveletTransform&PowerSpectrum.csv')

Y0_main = convert_y_dimensions(read_convert_output('label0.csv'))
Y1_main = convert_y_dimensions(read_convert_output('label1.csv'))

X = np.array(dataframe)
X = np.expand_dims(X, axis=2)

cnn_valence = get_model()
cnn_arousal = get_model()

cnn_valence.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.RMSprop(),
                    metrics=['acc'])

cnn_arousal.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.RMSprop(),
                    metrics=['acc'])


X0_train, X0_test, Y0_train, Y0_test = train_test_split(X, Y0_main, random_state=5, test_size=0.1)

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y1_main, random_state=5, test_size=0.1)

cnn_valence.fit(X0_train, Y0_train, epochs=100)
cnn_arousal.fit(X1_train, Y1_train, epochs=100)

print('Valence Train: \n' + str(cnn_valence.evaluate(X0_train, Y0_train)))
print('Valence Test: \n' + str(cnn_valence.evaluate(X0_test, Y0_test)))

print('Arousal Train: \n' + str(cnn_arousal.evaluate(X1_train, Y1_train)))
print('Arousal Test: \n' + str(cnn_arousal.evaluate(X1_test, Y1_test)))
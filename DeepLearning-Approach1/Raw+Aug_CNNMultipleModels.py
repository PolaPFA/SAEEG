import tensorflow as tf
import pandas as pd
import numpy as np
import csv
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

data_path = "E:\\College\\Graduation Project\\Dataset\\DEAP Dataset\\data_preprocessed_python\\data\\augmentedData\\"
log_path = "Logs\\"
data_to_be_read = 3

def get_model():
    model = tf.keras.models.Sequential()
    #model.add(tf.keras.layers.BatchNormalization(input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    #model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    #model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    #model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(5, 5)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.Dropout(0.6))
    #model.add(tf.keras.layers.Dense(40, activation='relu'))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    #model.add(tf.keras.layers.Dense(1))
    #model.add(tf.keras.layers.Activation(tf.keras.activations.sigmoid))
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
                data.append([1,0])
            else:
                data.append([0,1])
    data=np.array(data)
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

def read_input(start, end):
    output_data = []
    for i in range(start, end):
        sess_data = []
        for j in range(1,13):
            name = data_path + 'sess' + str(i).zfill(4) + '_electrode' + str(j).zfill(2) + '.csv'
            print(name)
            itr = 0
            with open(name) as f:
                data = csv.DictReader(f)
                electrode = []
                for row in data:
                    temp = list(row.values())
                    temp = [float(i) for i in temp]
                    electrode.append(temp.copy())
                    itr += 1
                    if itr % (data_to_be_read+1) == 0:
                        break
                sess_data.append(electrode)
        output_data.append(sess_data)
    return output_data

def initialize_models(input_model, number_of_models):
    valence_models = []
    arousal_models = []

    for model in range(number_of_models):
        current_model_0 = tf.keras.models.clone_model(input_model)
        current_model_0.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                                optimizer=tf.keras.optimizers.RMSprop(),
                                metrics=['acc'])
        valence_models.append(current_model_0)

        current_model_1 = tf.keras.models.clone_model(input_model)
        current_model_1.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                                optimizer=tf.keras.optimizers.RMSprop(),
                                metrics=['acc'])
        arousal_models.append(current_model_1)

    return valence_models, arousal_models

test_x_0 = []
test_x_1 = []
test_y_0 = []
test_y_1 = []


input_shape = (12, 8064, 4)


print('Getting models')
cnn_model = get_model()
valences, arousals = initialize_models(cnn_model, 32)

print('Before reading Logits')
Y0_main = read_convert_output('label0.csv')
Y1_main = read_convert_output('label1.csv')


step = 40
logFile = log_path + 'FinalRaw+Aug_CNNModelsNormalize.txt'
with open(logFile, 'w') as logf:
    for i in range(1, 33):
        #end = 41
        start = (i-1) * step + 1
        end = i * step + 1
        print('Before reading input')
        X = np.array(read_input(start, end))
        X_norm = np.zeros(X.shape)
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                X_norm[j, k, :, :] = normalize(X[j, k, :, :], axis=0, copy=True)

        X = np.transpose(X_norm, (0, 1, 3, 2))



        print('Before splitting')
        X0_train, X0_test, Y0_train, Y0_test = train_test_split(X, Y0_main[start-1: end-1],
                                                                random_state=1, test_size=0.2,
                                                                stratify=Y0_main[start-1: end-1])

        X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y1_main[start-1: end-1],
                                                                random_state=1, test_size=0.2,
                                                                stratify=Y0_main[start-1: end-1])


        # train_y0 = convert_y_dimensions(Y0_train)
        # train_y1 = convert_y_dimensions(Y0_train)
        # test_y1 = convert_y_dimensions(Y1_test)
        # test_y0 = convert_y_dimensions(Y0_test)

        # test_x_0.append(X0_test)
        # test_x_1.append(X1_test)
        # test_y_0.append(Y0_test)
        # test_y_1.append(Y1_test)

        print(X0_train.shape, X0_test.shape, Y0_train.shape, Y0_test.shape)
        print(X1_train.shape, X1_test.shape, Y1_train.shape, Y1_test.shape)

        #callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

        print('Training...')
        valences[i-1].fit(X0_train, Y0_train, epochs=50, batch_size=10,
                          validation_data=(X0_test, Y0_test))
        arousals[i-1].fit(X1_train, Y1_train, epochs=50, batch_size=10,
                          validation_data=(X1_test, Y1_test))
        logf.write('Person ' + str(i) + '\n')
        logf.write("TrainVal = " + str(valences[i-1].evaluate(X0_train, Y0_train)) +
                   " - TestVal = " + str(valences[i-1].evaluate(X0_test, Y0_test)) + '\n')
        logf.write("TrainAro = " + str(arousals[i-1].evaluate(X0_train, Y0_train)) +
                   " - TestVal = " + str(arousals[i-1].evaluate(X0_test, Y0_test)) + '\n')



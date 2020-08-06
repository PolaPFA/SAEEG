import tensorflow as tf
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split

data_path = "E:\\College\\Graduation Project\\Dataset\\DEAP Dataset\\data_preprocessed_python\\data\\augmentedData\\"
data_to_be_read = 3

def get_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    #model.add(tf.keras.layers.MaxPool2D())
    #model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3)))
    #model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    #model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 3)))
    #model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    #model.add(tf.keras.layers.MaxPool2D())
    #model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5)))
    #model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2))
    #model.add(tf.keras.layers.Dense(1))
    #model.add(tf.keras.layers.Activation(tf.keras.activations.sigmoid))
    model.add(tf.keras.layers.Softmax(axis=-1))
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

test_x_0 = []
test_x_1 = []
test_y_0 = []
test_y_1 = []

scale = 11
session_per_person = 40

input_shape = (4, 8064, 12)


print('Getting models')
cnn_valence_model = get_model()
cnn_valence_model.build((None, input_shape))
cnn_valence_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                          optimizer=tf.keras.optimizers.RMSprop(),
                          metrics=['acc'])

cnn_arousal_model = get_model()
cnn_arousal_model.build((None, input_shape))
cnn_arousal_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                          optimizer=tf.keras.optimizers.RMSprop(),
                          metrics=['acc'])

print('Before reading Logits')
Y0_main = convert_y_dimensions(read_convert_output('label0.csv'))
Y1_main = convert_y_dimensions(read_convert_output('label1.csv'))

start = 1
step = 160
for i in range(1, 9):
    end = i * step + 1
    print('Before reading input')
    X = np.array(read_input(start, end))
    X = np.transpose(X, (0, 2, 3, 1))

    print('Before splitting')
    X0_train, X0_test, Y0_train, Y0_test = train_test_split(X, Y0_main[start-1: end-1], shuffle=True,
                                                            random_state=32, test_size=0.2)

    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y1_main[start-1: end-1], shuffle=True,
                                                            random_state=32, test_size=0.2)


    #train_y0 = convert_y_dimensions(Y0_train)
    #train_y1 = convert_y_dimensions(Y0_train)
    #test_y1 = convert_y_dimensions(Y1_test)
    #test_y0 = convert_y_dimensions(Y0_test)

    test_x_0.append(X0_test)
    test_x_1.append(X1_test)
    test_y_0.append(Y0_test)
    test_y_1.append(Y1_test)

    print(X0_train.shape, X0_test.shape, Y0_train.shape, Y0_test.shape)
    print(X1_train.shape, X1_test.shape, Y1_train.shape, Y1_test.shape)

    print('Training...')
    cnn_valence_model.fit(X0_train, Y0_train, epochs=300)
    cnn_arousal_model.fit(X1_train, Y1_train, epochs=300)
    start = end
    print('Valence Train: ', cnn_valence_model.evaluate(X0_train, Y0_train))
    print('Arousal Train: ', cnn_arousal_model.evaluate(X1_train, Y1_train))


#test_x_0 = np.transpose(test_x_0, (0, 2, 3, 1))
#test_x_1 = np.transpose(test_x_1, (0, 2, 3, 1))
#print('Valence models:')
#print('Train: ', cnn_valence_model.evaluate(X0_train, train_y0))
print('Valence Test: ', cnn_valence_model.evaluate(test_x_0, test_y_0))

#print('Arousal models:')
print('Arousal Test: ', cnn_arousal_model.evaluate(test_x_1, test_y_1))
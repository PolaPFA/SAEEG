import tensorflow as tf
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_path = "G:\\Datasets&GP\\DEAP\\data_preprocessed_python\\convertedData\\\Frontal\\"
labelpath="G:\\Datasets&GP\\DEAP\\data_preprocessed_python\\convertedData\\"
def get_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.BatchNormalization( input_shape=(672, 1)))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=3))
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
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2,activation='softmax'))

    return model

def read_convert_output(file_name):
    data = []
    with open(labelpath+file_name) as f:
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



def read_convert_outputCat(file_name):
    data = []


    with open(labelpath+file_name) as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            temp = list(row.values())
            temp = [float(i) for i in temp]
            temp2 = temp[0]
            if temp2 >= 5:
                data.append((1,0))
            else:
                data.append((0,1))

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
                temp = np.array_split(temp, 12)
                temprow.append(temp.copy())
                if itr % 12 == 0:
                    data.append(temprow.copy())
                    temprow.clear()
                itr += 1
    data=np.array(data)

    return data

def convert_x_dimensions(input,i,x):
    temp = input[:, i, x, :]
    current_x = np.expand_dims(temp, axis=1)
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
        current_model_0.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                                optimizer=tf.keras.optimizers.SGD(),
                                metrics=['acc'])
        valence_models.append(current_model_0)

        current_model_1 = tf.keras.models.clone_model(cnn_model)
        current_model_1.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                                optimizer=tf.keras.optimizers.SGD(),
                                metrics=['acc'])
        arousal_models.append(current_model_1)
    return valence_models, arousal_models


def maxvotesoftmax(list):
    output = []

    list=np.transpose(list,(1,0,2))
    for index in list:
        onecount=0
        zerocount=0
        for element in index:
            if element[0]==1 and element[1] ==0:
                onecount += 1
            elif element[0]==0 and element[1] ==1:
                zerocount +=1
        if onecount > zerocount:
            output.append((1,0))
        else:
            output.append((0,1))
    return output


print('Before reading data')

X = read_input_data()

print('Before reading logits')
Y0 = read_convert_outputCat('label0.csv')
Y1 = read_convert_outputCat('label1.csv')



print('Before getting model')
cnn_model = get_model()

print('Before initializing models')
valence_models, arousal_models = initialize_models(12)

print('Before splitting')



X0_train, X0_test, Y0_train, Y0_test = train_test_split(X, Y0, shuffle=False, random_state=32, test_size=0.4)

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, Y1, shuffle=False, random_state=32, test_size=0.4)






for i in range(12):
    train_y1 = convert_y_dimensions(Y1_train[i])
    train_y0 = convert_y_dimensions(Y0_train[i])
    for x in range (12):
        print('Valancey  Training Electrode : ', i + 1)
        print('Time Sample  Training  ', x + 1)
        train_x0 = convert_x_dimensions(X0_train,i,x)
        train_x1 = convert_x_dimensions(X1_train, i, x)
        valence_models[i].fit(train_x0, Y0_train, epochs=1,batch_size=50)
        print("______________________________________________________________________________________________________")

        print('Arrosal  Training Electrode : ', i + 1)
        print('Time Sample  Training  ', x + 1)


        arousal_models[x].fit(train_x1, Y1_train, epochs=1,batch_size=50)
        print("______________________________________________________________________________________________________")





def getting_finaloutputsoftmax(list):
    out=[]
    list=np.array(list)
    for item in list:
        if item[1] ==0 and item[0]==1:
            out.append((1,0))
        else:
            out.append((0,1))
    return  out

#Testing the double models:
valencies = []
arousals = []

for i in range(12):
    valenciesout=[]
    arousalsout=[]

    for x in range(12):
        test_x0 = convert_x_dimensions(X0_test, i, x)
        test_x1 = convert_x_dimensions(X1_test, i, x)
        valenciesout.append(getting_finaloutputsoftmax(valence_models[i].predict(test_x0)))
        arousalsout.append(getting_finaloutputsoftmax(arousal_models[i].predict(test_x1)))

    valencies.append(maxvotesoftmax(valenciesout))
    arousals.append(maxvotesoftmax(arousalsout))

valencies = maxvotesoftmax(valencies)
arousals = maxvotesoftmax(arousals)
print("Validation test:")
print('Valence accuracy:', accuracy_score(valencies, Y0_test))

print('Arousals accuracy:', accuracy_score(arousals, Y1_test))





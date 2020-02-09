import pickle
import pandas

#Ser the dataset path here (.dat) files
mypath = "E:\\College\\Graduation Project\\Dataset\\DEAP Dataset\\data_preprocessed_python\\data\\"

#Converting each file to csv file
for i in range(1,33):
    name = ""
    if i < 10:
        name = "s0" + str(i)
    else:
        name = "s" + str(i)
    print(name)
    f = open(mypath+name+".dat", 'rb')
    data = pickle.load(f, encoding='latin1')
    labels = data["labels"]
    for k in range(40):
        X = data["data"][k][:][:]
        X = pandas.DataFrame(X)
        X.to_csv(mypath+'convertedData\\'+name+'.csv', mode='a', index=False)
        for j in range(4):
            Y = labels[k][j]
            Y = pandas.Series(Y)
            Y.to_csv(mypath + 'convertedData\\label' + str(j) + '.csv', mode='a', index=False)
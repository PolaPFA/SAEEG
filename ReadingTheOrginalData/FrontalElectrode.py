import pickle
import pandas

#Ser the dataset path here (.dat) files
mypath = "G:\\Datasets&GP\\DEAP\\data_preprocessed_python\\"

#Converting each file to csv file
for i in range(1,33):
    t=0
    name = ""
    if i < 10:
        name = "s0" + str(i)
    else:
        name = "s" + str(i)
    print(name)
    f = open(mypath+name+".dat", 'rb')
    data = pickle.load(f, encoding='latin1')
    labels = data["labels"]
    electrode =[0,1,2,3,4,5,24,25,26,27,28,29]
    for k in electrode:
        X = data["data"][k][:][:]
        X = pandas.DataFrame(X)
        if t==0:
            X.to_csv(mypath+'convertedData\\\Frontal\\'+name+"Frontal"+'.csv', mode='a', index=False)
            t+=1
        else:
            X.to_csv(mypath + 'convertedData\\\Frontal\\' + name+"Frontal" + '.csv', mode='a', index=False,header= False)



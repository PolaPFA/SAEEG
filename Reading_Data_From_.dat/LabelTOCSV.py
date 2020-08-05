import pickle
import pandas


mypath = "G:\\Datasets&GP\\DEAP\\data_preprocessed_python\\"
#mypath = "E:\\College\\Graduation Project\\Dataset\\DEAP Dataset\\data_preprocessed_python\\data\\"

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
    #no of electrode

        #Labels
    for k in range(40)  :

        for j in range(4):
            Y = labels[k][j]
            Y = pandas.Series(Y)

            if i==0:
                Y.to_csv(mypath+'convertedData\\label'+str(j)+'.csv', mode='a', index=False)
            else:
                Y.to_csv(mypath+'convertedData\\label'+str(j)+'.csv', mode='a', index=False,header=False)
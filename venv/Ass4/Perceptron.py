from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn import preprocessing

class Perceptron:
    def __init__(self, data, a=0.1, maxcounts=100, cv=10):

        self.data = data
        self.a = a
        self.maxcounts = maxcounts
        self.cv = cv



    #data is already normalized
    def normalizationTrain(self, traindata):
        norm = preprocessing.scale(traindata)
        return norm,norm.mean(axis=0),norm.std(axis=0)



    def importData(self,fileLocation):
        dataSet = pd.read_csv(filepath_or_buffer=fileLocation, header=None)
        dataSet = dataSet.drop_duplicates()
        return dataSet

    def splitData(self,df,cv):
        df = df.sample(frac = 1)
        row = df.shape[0]
        train = int(row*(cv-1)/cv)
        trainData = df.iloc[:train,:(df.shape[1]-1)]
        trainTarget = df.iloc[:train,(df.shape[1]-1):]
        testData = df.iloc[train:,:df.shape[1]-1]
        testTarget= df.iloc[train:,(df.shape[1]-1):]


        return trainData,trainTarget,testData,testTarget


    def addConstant(self,data):
        data.insert(data.shape[1],data.shape[1],1)
        return data


    def cost(self,data,target,w):

        p =data@w.transpose()

        target.columns = [p.columns[0]]

        err = p.add(target*(-1), fill_value=0)
        rms = math.sqrt(np.sum(err** 2)/data.shape[0])

        er = err.transpose()@data
        return rms,er


    def grad(self,data,target,w,a):
        eresult = []
        count = 0
        error = 0
        while self.notconverged(count,error):
            w,error = self.fit(data,target,w)

            eresult.append(error)
            count+=1
        return w,eresult






    def notconverged(self,count,error):
        if count == 0: return True
        if np.sum(error) == 0: return False
        if count >=self.maxcounts: return False

        return True

    def validate(self):
        w = np.matrix(np.zeros(d.shape[1]-1))
        data,target,testd,testt = self.splitData(self.data,self.cv)
        dt,mean, std = self.normalizationTrain(data)

        dataTest,mean,std = self.normalizationTrain(testd)

        wmin,jresult = self.grad(dt,target,w,self.a)


        testrms = self.predict(dataTest,wmin)
        accuracy,precision, recall = self.calculateClassMetrics(testt.values,testrms)

        return accuracy,precision,recall

    def test_normalize(X, mean, std):
        X_norm = (X - mean) / std
        X_norm = add_x0(X_norm)

        return X_norm
    def predict(self, data, w):
        pre = data@w.transpose()

        pre = [[1 if x > 0 else -1] for x in pre]

        return pre

    def fit(self, data, target, w):
        error = target.values - self.predict(data, w)

        w += self.a * (error.transpose() @ data)
        return w, error

    def calculateClassMetrics(self, testClasses, prediction):


        accuracy = accuracy_score(testClasses, prediction)
        precision = precision_score(testClasses, prediction, average='weighted')
        recall = recall_score(testClasses, prediction, average='weighted')

        return accuracy, precision, recall

    def foldup(self):


        a = []
        p = []
        r = []
        for i in range(self.cv):
            accuracy,precision,recall = self.validate()
            a.append(accuracy)
            p.append(precision)
            r.append(recall)
        return np.mean(a),np.std(a),np.mean(p),np.std(p),np.mean(r),np.std(r)





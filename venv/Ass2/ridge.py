from numpy.linalg import inv
import pandas as pd
import math
import numpy as np

import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


class ridge:
    def __init__(self,p,c,cv=10):
        self.p = p
        self.c = c
        self.cv = cv
    def importData(self, fileLocation):
        dataSet = pd.read_csv(filepath_or_buffer=fileLocation, header=None)
        dataSet = dataSet.drop_duplicates()
        return dataSet

    def processData(self, data, target, p):
        datanew = data.copy()
        for i in range(1, p, 1):
            temp = data ** (i + 1)
            temp.columns = np.arange(((i) * data.shape[1]), ((i + 1) * data.shape[1]), 1)
            datanew = datanew.join(temp)

        return datanew, target

    def splitData(self, df, cv):
        df = df.sample(frac=1)
        row = df.shape[0]
        if cv == 0:
            train = df.shape[0]
        else:
            train = int(row * (cv - 1) / cv)

        trainData = df.iloc[:train, :(df.shape[1] - 1)]
        trainTarget = df.iloc[:train, (df.shape[1] - 1):]
        testData = df.iloc[train:, :df.shape[1] - 1]
        testTarget = df.iloc[train:, (df.shape[1] - 1):]

        return trainData, trainTarget, testData, testTarget

    def normalEquation(self, X, y):

        w = np.dot(
            np.dot(inv(np.dot(X.transpose(), X)), X.transpose()), y)
        w = pd.DataFrame(data=w)
        return w

    def predictError(self, data, target, w):
        p = data @ w

        target.columns = [p.columns[0]]

        err = p.add(target * (-1), fill_value=0)
        rms =math.sqrt(np.sum(err ** 2) / data.shape[0])

        return np.sum(err ** 2) / data.shape[0],rms

    def normalizationTrain(self,data):
        mean = {}
        datanew = data.copy()

        for column in data:
            mean[column] = data[column].mean()


        for index, row in data.iterrows():
            for label, content in row.iteritems():
                nor = (content - mean[label])

                datanew.loc[index, label] = nor

        return datanew,mean

    def normalizationTest(self,data,mean):

        datanew = data.copy()
        for index, row in data.iterrows():
            for label,content in row.iteritems():
                nor = (content - mean[label])

                datanew.loc[index, label] = nor

        return datanew
    def ridgeRegression(self, data,target,c):

        attributes = data.shape[1]
        lamda = np.dot(c, np.identity(attributes))
        tt = np.dot(data.transpose(),data)
        verse = inv(tt+lamda)
        t = np.dot(verse,data.transpose())


        w = np.dot(t,target)
        w = pd.DataFrame(data = w)


        return w

    def foldup(self, data):

        for i in range(1,(self.p+1),1):
            cTrainRMSE = []
            cTestRMSE = []
            for b in range(0,(self.c*10),2):
                c = b/10
                trainRMSE = []
                testRMSE = []
                traindata,traintarget, testdata,testtarget = self.splitData(data,self.cv)
                traindata,traintarget = self.processData(traindata,traintarget,i)
                testdata,testtarget = self.processData(testdata,testtarget,i)
                traindata,mean = self.normalizationTrain(traindata)
                traintarget,meantarget = self.normalizationTrain(traintarget)
                testtarget = self.normalizationTest(testtarget,meantarget)
                testdata = self.normalizationTest(testdata,mean)
                w = self.ridgeRegression(traindata,traintarget,c)
                testsse,testrmse = self.predictError(testdata,testtarget,w)
                trainsse, trainrmse = self.predictError(traindata, traintarget, w)
                trainRMSE.append(trainrmse)
                testRMSE.append(testrmse)
                cTrainRMSE.append(np.mean(trainRMSE))
                cTestRMSE.append(np.mean(testRMSE))
            plt.plot(np.arange(0,self.c,0.2),cTrainRMSE, label='Training Data Set')
            plt.plot(np.arange(0,self.c,0.2), cTestRMSE, label='Test Data Set')
            plt.xlabel('c')
            plt.ylabel('Mean RMSE')
            plt.show()
        return cTrainRMSE, cTestRMSE, np.mean(cTrainRMSE), np.mean(cTestRMSE)



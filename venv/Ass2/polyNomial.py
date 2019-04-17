from numpy.linalg import inv
import pandas as pd
import math
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

class poly:
    def __init__(self,p):
        self.p = p

    def importData(self, fileLocation):
        dataSet = pd.read_csv(filepath_or_buffer=fileLocation, header=None)
        dataSet = dataSet.drop_duplicates()
        return dataSet



    def processData(self,data,target,p):
        datanew =data.copy()
        for i in range(1,p,1):
            temp = data ** (i+1)
            temp.columns=np.arange(((i)*data.shape[1]),((i+1)*data.shape[1]),1)
            datanew =datanew.join(temp)


        datanew[datanew.shape[1]] = 1
        return datanew,target

    def splitData(self, df, cv):
        df = df.sample(frac=1)
        row = df.shape[0]
        if cv==0:
            train =df.shape[0]
        else:
            train = int(row * (cv - 1) / cv)

        trainData = df.iloc[:train, :(df.shape[1] - 1)]
        trainTarget = df.iloc[:train, (df.shape[1] - 1):]
        testData = df.iloc[train:, :df.shape[1] - 1]
        testTarget = df.iloc[train:, (df.shape[1] - 1):]

        return trainData, trainTarget, testData, testTarget


    def normalEquation(self, X,y):

        w = np.dot(
            np.dot(inv(np.dot(X.transpose(), X)), X.transpose()), y)
        w = pd.DataFrame(data = w)
        return w




    def predictError(self, data, target, w):

        p = data @w

        target.columns = [p.columns[0]]

        err = p.add(target * (-1), fill_value=0)
        rms = math.sqrt(np.sum(err ** 2) / data.shape[0])

        return rms
    def foldu(self, data, valid,p):
        rmse=[]
        rmset=[]
        for j in range(1,p+1,1):
            traindata,traintarget, tt,dd = self.splitData(data,0)
            testdata, testtarget, uu, qq = self.splitData(valid, 0)
            traindata,traintarget = self.processData(traindata,traintarget,j)
            testdata,testtarget = self.processData(testdata,testtarget,j)
            w = self.normalEquation(traindata, traintarget)
            rmstrain = self.predictError(traindata, traintarget, w)
            rmstest = self.predictError(testdata, testtarget, w)
            print(testdata)
            rmse.append(rmstrain)
            rmset.append(rmstest)
        plt.plot(np.arange(1, (self.p + 1), 1), rmse, label='Test Data Set')
        plt.plot(np.arange(1, (self.p+1), 1), rmset, label='Test Data Set')
        plt.xlabel('c')
        plt.ylabel('Mean RMSE')
        plt.show()

        return rmstest, rmstrain,np.mean(rmstest),np.mean(rmstrain)


    def foldup(self, data, cv,p):
        rmse=[]
        rmset=[]
        for j in range(1,p+1,1):
            rms=[]
            rmst = []
            for i in range(cv):
                traindata,traintarget, testdata,testtarget = self.splitData(data,cv)
                traindata,traintarget = self.processData(traindata,traintarget,j)
                testdata,testtarget = self.processData(testdata,testtarget,j)
                w = self.normalEquation(traindata, traintarget)
                rmstrain = self.predictError(traindata, traintarget, w)
                rmstest = self.predictError(testdata, testtarget, w)

                rms.append(rmstrain)
                rmst.append(rmstest)
            rmse.append(np.mean(rms))
            rmset.append(np.mean(rmst))
        plt.plot(np.arange(1, (self.p+1), 1), rmse, label='Training Data Set')
        plt.plot(np.arange(1, (self.p+1), 1), rmset, label='Test Data Set')
        plt.xlabel('p')
        plt.ylabel('Mean RMSE')
        plt.show()

        return rmstest, rmstrain,np.mean(rmstest),np.mean(rmstrain)




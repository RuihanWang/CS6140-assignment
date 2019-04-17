from numpy.linalg import inv
import pandas as pd
import math
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

class NormalEquation:

    def splitData(self, df, cv):
        df = df.sample(frac=1)
        row = df.shape[0]
        train = int(row * (cv - 1) / cv)
        trainData = df.iloc[:train, :(df.shape[1] - 1)]
        trainData[trainData.shape[1]] = 1
        trainTarget = df.iloc[:train, (df.shape[1] - 1):]
        testData = df.iloc[train:, :df.shape[1] - 1]
        testData[testData.shape[1]] = 1
        testTarget = df.iloc[train:, (df.shape[1] - 1):]

        return trainData, trainTarget, testData, testTarget
    def importData(self, fileLocation):
        dataSet = pd.read_csv(filepath_or_buffer=fileLocation, header=None)
        dataSet = dataSet.drop_duplicates()
        return dataSet



    def predictError(self, data, target, w):
        p = data @w

        target.columns = [p.columns[0]]

        err = p.add(target * (-1), fill_value=0)
        rms = math.sqrt(np.sum(err ** 2) / data.shape[0])

        return rms

    def normalEquation(self, X,y):

        w = np.dot(
            np.dot(inv(np.dot(X.transpose(), X)), X.transpose()), y)
        w = pd.DataFrame(data = w)
        return w
    def foldup(self, data, cv):
        rmstest = []
        rmstrain = []
        for i in range(cv):
            traindata,traintarget, testdata,testtarget = self.splitData(data,cv)
            w = self.normalEquation(traindata,traintarget)
            rmstest.append(self.predictError(testdata, testtarget,w))
            rmstrain.append(self.predictError(traindata, traintarget,w))

        return rmstest, rmstrain,np.mean(rmstest),np.std(rmstest),np.mean(rmstrain),np.std(rmstrain)


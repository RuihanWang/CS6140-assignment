
from sklearn.metrics import accuracy_score,precision_score,recall_score
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
from sklearn import preprocessing
import numpy as np


class PerceptronKernel:
    def __init__(self, dataSet, maxrounds=100, linear=True, cv=10):
        self.dataSet = dataSet
        self.maxrounds = maxrounds
        self.cv = cv
        self.linear = linear

    def splitK(self, df, cv=10):
        df = df.sample(frac=1)
        row = df.shape[0]
        train = int(row * (cv - 1) / cv)
        traindata = df.iloc[:train, :]

        testdata = df.iloc[train:, :]

        return traindata, testdata

    def splitm(self,df,m=10):
        df = df.sample(frac = 1)
        row = df.shape[0]
        train = int(row*(m-1)/m)
        trainData = df.iloc[:train,:-1]
        trainData = self.addConstant(trainData)
        trainData = np.matrix(trainData)
        trainTarget = (df.iloc[:train,-1].T.values)
        testData = df.iloc[train:, :-1]
        testData = self.addConstant(testData)
        testData = np.matrix(testData)
        testTarget= df.iloc[train:,-1].T.values


        return trainData,trainTarget,testData,testTarget

    def splittarget(self, df):
        data = np.matrix(df.iloc[:, :-1])
        target = np.matrix(df.iloc[:, -1].T)
        return data, target

    def calculateClassMetrics(self, testClasses, prediction):

        accuracy = accuracyqscore(testClasses, prediction)
        precision = precisionqscore(testClasses, prediction, average='weighted')
        recall = recallqscore(testClasses, prediction, average='weighted')

        return accuracy, precision, recall

    def addConstant(self, data):
        data.insert(data.shape[1], data.shape[1], 1)
        return data

    # gs = [0.01,0.05,0.10,0.20,0.25]
    def rbfKernel(self, xJ, xI, gamma=0.1):
        return np.exp(-gamma * np.linalg.norm(xJ - xI)**2)

    def grad(self, data,target):


        weights = np.matrix(np.zeros(data.shape[0])).T
        ks = [[self.rbfKernel(data[j], data[i])for j in range(data.shape[0])] for i in range(data.shape[0])]
        maxrounds = 0



        for q in range(self.maxrounds):
            maxrounds = q + 1
            loopError = 0
            for i in range(data.shape[0]):
                sumVal = 0
                for j in range(data.shape[0]):
                    sumVal += weights[j] * target[j] * ks[i][j]

                predictedVal = 1 if sumVal >= 0.0 else -1
                if target[i] != predictedVal:
                    weights[i] += 1
                    loopError += 1
            if loopError == 0:
                break

        return maxrounds, weights

    def accuracy(self, traindata,traintarget,testdata,testtarget, weights):

        prediction = []
        for i in range(testdata.shape[0]):
            sumVal = 0
            for j in range(traindata.shape[0]):
                kernel = self.rbfKernel(
                        traindata[j], testdata[i])

                sumVal += weights[j] * traintarget[j] * kernel
            predictedVal = 1 if sumVal >= 0.0 else -1
            prediction.append(predictedVal)

        accuracy = accuracy_score(testtarget, prediction)

        return accuracy

    def foldup(self):
        maxrounds = []
        accuracies = []

        fold = 1

        for i in range (self.cv):
            print(i)
            traindata,traintarget,testdata,testtarget = self.splitm(self.dataSet)
            count, weights = self.grad(traindata,traintarget)
            maxrounds.append(count)

            accuracy = self.accuracy(traindata,traintarget,testdata,testtarget, weights)
            accuracies.append(accuracy)

            print(i, count, accuracy)
            fold += 1

        print(np.mean(maxrounds), np.mean(accuracies))

        print(np.std(maxrounds), np.std(accuracies))




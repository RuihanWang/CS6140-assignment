from __future__ import division
import math
import Tree
import pandas as pd
import numpy as np


class regression:
    def __init__(self, data, min,cv=10):
        self.data = data
        self.cv = cv
        self.min = min

    def normalizationTrain(self,data):

        for column in data:
            max = np.amax(data[column])
            min = np.amin(data[column])
            data[column] = (data[column] - min) / (max - min)

        return data

    def splitData(self, df, cv):

        df = df.sample(frac=1)
        row = df.shape[0]
        train = int(row * (cv - 1) / cv)
        trainData = df.iloc[:train,:]
        testData = df.iloc[train:, :df.shape[1] - 1]
        testTarget = df.iloc[train:, (df.shape[1] - 1):]

        return trainData, testData, testTarget

    def SD(self, dt):

        target= dt.iloc[:,(dt.shape[1]-1):]
        sd = np.sum((target-target.mean())**2)


        return sd

    def BinarysplitData(self,data, attribute, value):
        split = []

        data1 = data[data[attribute] < value]
        data2 = data[data[attribute] >= value]
        split.append(data1)
        split.append(data2)



        return split
    def getSDgain(self,data,attribute,value):
        sd = self.SD(data)
        split = self.BinarysplitData(data,attribute,value)
        sdnew = 0.0
        for dt in split:
            sdnew += self.SD(dt)

        gain = sd - sdnew
        return gain


    def getBestSplit(self, data):
        gain = 0
        bestAttribute=""
        bestValue=0.0
        attributes = list(data.columns)
        attributes = attributes[:-1]
        for attribute in attributes:
            for row in data[attribute]:
                newgain = float(self.getSDgain(data,attribute,row))

                if newgain>=gain:
                    gain = newgain
                    bestAttribute = attribute
                    bestValue = row



        return bestAttribute, bestValue

    def isPure(self, data):
        return data.iloc[:,-1].unique().shape[0] == 1
    def keepSplit(self,data,min):
        if data.shape[0]>=min:
            return True
        return False

    def buildRegressionTree(self, data, min):
        root  =  Tree.BinarySplitTree()
        if self.keepSplit(data,min):
            bestAttribute,bestValue = self.getBestSplit(data)
            split = self.BinarysplitData(data,bestAttribute,bestValue)
            child1=split[0]
            child2=split[1]
            root.leftchild = self.buildRegressionTree(child1,min)
            root.rightchild = self.buildRegressionTree(child2, min)
            root.value = bestValue
            root.attribute = bestAttribute
        else:
            root.isleaf = True
            root.truevalue = np.mean(data.iloc[:,-1])



        return root


    def predict(self,testdata,tree):
        prediction = 0
        if tree.isleaf == True:
            prediction = tree.truevalue
            return prediction

        if testdata[tree.attribute] < tree.value:
            prediction = self.predict(testdata,tree.leftchild)

        else:
            prediction = self.predict(testdata,tree.rightchild)
        return prediction
    def predictfoldup(self,data,tree):
        pre = []
        for index,row in data.iterrows():
            p = self.predict(row,tree)
            pre.append(p)

        return pre


    def SSE(self,prediction,testlabel):

        sse = np.sum((np.array(prediction)-np.array(testlabel).transpose())**2)



        return sse
    def foldup(self):
        data = self.data

        data = pd.read_csv(data, header=None)


        dt = pd.DataFrame(data=data)
        dt = dt.drop_duplicates()
        mins = [data.shape[0] * 0.05,data.shape[0] * 0.10,data.shape[0] * 0.15,data.shape[0] * 0.20]



        for min in mins:


            SSE = []
            trainSSE= []
            tree = self.buildRegressionTree(dt, min)
            for i in range(1, self.cv + 1):
                traindata, testdata, testtarget = self.splitData(dt, self.cv)
                testlabel = testtarget.values
                train = traindata.iloc[:, :(traindata.shape[1] - 1)]
                trainlabel = traindata.iloc[:, -1].tolist()
                trainprediction = self.predictfoldup(train, tree)
                prediction = self.predictfoldup(testdata, tree)
                trainSE = self.SSE(trainprediction,trainlabel)
                SE = self.SSE(prediction, testlabel)
                SSE.append(SE / testdata.shape[0])
                trainSSE.append(trainSE/traindata.shape[0])


            print(min,' ', np.mean(SSE))
            print(min, ' ', np.std(SSE))
            print(min,' train',np.mean(trainSSE))
            print(min,' train',np.std(trainSSE))





from __future__ import division
import math
import Tree
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

class MultiDecisionTree:
    def __init__(self, data, cv=10):
        self.data = data
        self.cv = cv

    def normalizationTrain(self,data):

        for column in data:
            max = np.amax(data[column])
            min = np.amin(data[column])
            data[column] = (data[column] - min) / (max - min)

        return data


    def MultisplitData(self,data,attribute):
        split = []
        values = []
        uniquevalue = data[attribute].unique()
        for v in uniquevalue:
            datanew = data[data[attribute] == v]
            split.append(datanew)
            values.append(v)

        return split,values



    def splitData(self, df, cv):
        df = df.sample(frac=1)
        row = df.shape[0]
        train = int(row * (cv - 1) / cv)
        trainData = df.iloc[:train,:]
        testData = df.iloc[train:, :]
        testTarget = df.iloc[train:, -1]

        return trainData, testData, testTarget


    def entropy(self,data):

        items = data.shape[0]
        c = data.columns[-1]

        d = data[c].value_counts()
        d = d / items
        entro = 0
        for e in d:
            entro -= e * np.log2(e)
        return entro



    def informationGain(self,data,attribute):
        entropy = self.entropy(data)

        split,values = self.MultisplitData(data,attribute)
        entropynew = 0.0
        for dt in split:
            entropynew += self.entropy(dt)*(dt.shape[0]/data.shape[0])

        informationgain = entropy - entropynew


        return informationgain

    def getBestSplit(self, data):
        gain = 0
        bestAttribute=""
        attributes = list(data.columns)
        attributes = attributes[:-1]
        bestValue = 0

        for attribute in attributes:
            newgain = float(self.informationGain(data,attribute))

            if newgain>gain:
                gain = newgain
                bestAttribute = attribute



        return bestAttribute, bestValue

    def isPure(self,data):
        return data.iloc[:, -1].unique().shape[0] == 1


    def keepSplit(self,data,min):
        if data.shape[0]<=min:
            return False
        if self.isPure(data):
            return False
        return True

    def buildRegressionTree(self, data, min):

        root  =  Tree.MultiSplitTree()
        if self.keepSplit(data,min):
            bestAttribute,bestValue = self.getBestSplit(data)
            split,values = self.MultisplitData(data,bestAttribute)


            for dt in split:
                root.child.append(self.buildRegressionTree(dt,min))

            for j in range(0,len(values)):
                root.child[j].value = values[j]


            root.attribute = bestAttribute
        else:
            root.isleaf = True

            root.truevalue = self.getMostLabel(data)


        return root
    def getMostLabel(self,data):
        return data[data.columns[-1]].value_counts().idxmax()


    def predict(self,testdata,tree):
        prediction = 0
        if tree.isleaf == True:
            prediction = tree.truevalue
            return prediction

        for i in range(0,len(tree.child)):

            if testdata[tree.attribute] == tree.child[i].value:
                prediction = self.predict(testdata, tree.child[i])





        return prediction
    def predictfoldup(self,data,tree):#no target value
        pre = []
        for index,row in data.iterrows():

            p = self.predict(row,tree)

            pre.append(p)
        return pre

    def confusionmatrix(self,prediction,testlabel):
        #print(prediction)
        #print(testlabel)

        return accuracy_score(testlabel, prediction),confusion_matrix(testlabel,prediction)



    def foldup(self):
        data = self.data
        data = pd.read_csv(data, header=None)
        dt = pd.DataFrame(data=data)
        dt = dt.drop_duplicates()
        mins = [dt.shape[0]*0.05,dt.shape[0]*0.10,dt.shape[0]*0.15]


        for min in mins:
            pre = []
            tree = self.buildRegressionTree(dt, min)
            con = []
            pretrain = []

            for i in range(1,self.cv+1,1):
                traindata,testdata,testtarget = self.splitData(dt,self.cv)
                testlabel = testtarget.transpose().tolist()

                train = traindata.iloc[:, :(traindata.shape[1] - 1)]
                trainlabel = traindata.iloc[:, -1].tolist()
                trainprediction = self.predictfoldup(train, tree)
                trainprecision, trainconfusionmatrix = self.confusionmatrix(trainprediction, trainlabel)
                pretrain.append(trainprecision)

                prediction = self.predictfoldup(testdata,tree)
                precision,confusionmatrix = self.confusionmatrix(prediction,testlabel)
                pre.append(precision)
                try:
                    con = np.add(con, confusionmatrix)

                except:
                    con = confusionmatrix
            print(min,' ',np.mean(pre))
            print(min,' ',np.std(pre))
            print(con)
            print(min,'trian',np.mean(pretrain))



from __future__ import division
import math
import Tree
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class decisionTree:
    def __init__(self, data, cv=10):
        self.data = data
        self.cv = cv


    def BinarysplitData(self,data, attribute, value ):
        split = []
        try:
            var = int(value)

            data1 = data[data[attribute] < value]
            data2 = data[data[attribute] >= value]
            split.append(data1)
            split.append(data2)

        except:
            data1 = data[data[attribute] == value]
            data2 = data[data[attribute] != value]
            split.append(data1)
            split.append(data2)

        return split



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



    def informationGain(self,data,attribute,value):
        entropy = self.entropy(data)

        split = self.BinarysplitData(data,attribute,value)
        entropynew = 0.0
        for dt in split:
            entropynew += self.entropy(dt)*(dt.shape[0]/data.shape[0])

        informationgain = entropy - entropynew


        return informationgain

    def getBestSplit(self, data):
        gain = 0
        bestAttribute=""
        bestValue=0.0
        attributes = list(data.columns)
        attributes = attributes[:-1]

        for attribute in attributes:
            uniquevalue = data[attribute].unique()
            for v in uniquevalue:
                newgain = float(self.informationGain(data,attribute,v))

                if newgain>gain:
                    gain = newgain
                    bestAttribute = attribute
                    bestValue = v



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
        root  =  Tree.BinarySplitTree()
        if self.keepSplit(data,min):
            bestAttribute,bestValue = self.getBestSplit(data)
            split = self.BinarysplitData(data,bestAttribute,bestValue)
            child1=split[0]
            child2 = split[1]
            root.leftchild = self.buildRegressionTree(child1,min)
            root.rightchild = self.buildRegressionTree(child2, min)
            root.value = bestValue
            root.attribute = bestAttribute
        else:
            root.isleaf = True

            root.truevalue = self.getMostLabel(data)


        return root
    def getMostLabel(self,data):
        return data[data.columns[-1]].value_counts().idxmax()


    def predict(self,testdata,tree):

        if tree.isleaf == True:
            prediction = tree.truevalue
            return prediction

        if testdata[tree.attribute] < tree.value:
            prediction = self.predict(testdata,tree.leftchild)

        else:
            prediction = self.predict(testdata,tree.rightchild)
        return prediction
    def predictc(self,testdata,tree):
        if tree.isleaf == True:
            prediction = tree.truevalue
            return prediction
        if testdata[tree.attribute] == tree.value:

            prediction = self.predict(testdata,tree.leftchild)

        else:
            prediction = self.predict(testdata,tree.rightchild)
        return prediction
    def predictfoldup(self,data,tree):#no target value
        pre = []
        for index,row in data.iterrows():

            p = self.predict(row,tree)

            pre.append(p)
        return pre


    def predictfoldupc(self, data, tree):  # no target value
        pre = []
        for index, row in data.iterrows():
            p = self.predictc(row, tree)

            pre.append(p)
        return pre

    def confusionmatrix(self,prediction,testlabel):
        #print(prediction)
        #print(testlabel)

        return accuracy_score(testlabel, prediction), confusion_matrix(testlabel,prediction)



    def foldup(self):
        if self.data == 'mushroom.csv':
            data = self.data
            data = pd.read_csv(data, header=None)
            dt = pd.DataFrame(data=data)
            dt = dt.drop_duplicates()
            mins = [dt.shape[0] * 0.05, dt.shape[0] * 0.10, dt.shape[0] * 0.15]


            for min in mins:
                pre = []
                confusion = []
                pretrain = []
                tree = self.buildRegressionTree(dt, min)

                for i in range(1, self.cv + 1, 1):
                    traindata, testdata, testtarget = self.splitData(dt, self.cv)
                    testlabel = testtarget.transpose().tolist()

                    train = traindata.iloc[:,:(traindata.shape[1]-1)]
                    trainlabel = traindata.iloc[:,-1].tolist()
                    trainprediction = self.predictfoldup(train, tree)
                    trainprecision, trainconfusionmatrix = self.confusionmatrix(trainprediction, trainlabel)
                    pretrain.append(trainprecision)




                    prediction = self.predictfoldupc(testdata, tree)
                    precision,confusionmatrix = self.confusionmatrix(prediction, testlabel)
                    pre.append(precision)
                    try:
                        confusion = np.add(confusion, confusionmatrix)

                    except:
                        confusion = confusionmatrix

                print(min, ' mushroom', np.mean(pre))
                print(min, 'mushroom ', np.std(pre))
                print(confusion)
                print(np.mean(pretrain))


        else:
            data = self.data
            data = pd.read_csv(data, header=None)
            dt = pd.DataFrame(data=data)
            dt = dt.drop_duplicates()
            mins = [dt.shape[0]*0.05,dt.shape[0]*0.10,dt.shape[0]*0.15]



            for min in mins:
                confusion = []
                tree = self.buildRegressionTree(dt, min)
                pre = []
                pretrain = []

                for i in range(1,self.cv+1,1):
                    traindata,testdata,testtarget = self.splitData(dt,self.cv)
                    testlabel = testtarget.transpose().tolist()



                    train = traindata.iloc[:,:(traindata.shape[1]-1)]
                    trainlabel = traindata.iloc[:,-1].tolist()
                    trainprediction = self.predictfoldup(train, tree)
                    trainprecision, trainconfusionmatrix = self.confusionmatrix(trainprediction, trainlabel)
                    pretrain.append(trainprecision)



                    prediction = self.predictfoldup(testdata,tree)
                    precision,confusionmatrix = self.confusionmatrix(prediction,testlabel)

                    pre.append(precision)

                    try:
                        confusion = np.add(confusion, confusionmatrix)

                    except:
                        confusion = confusionmatrix
                print(min,' ',np.mean(pre))
                print(min,' ',np.std(pre))
                print(min,'trainaccurayc',np.mean(pretrain))
                print(confusion)




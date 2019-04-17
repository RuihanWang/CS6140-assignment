from sklearn.svm import SVC
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.utils import shuffle
import warnings

class SVMMultiRBF:
    def __init__(self, dataSet, linear=True, cv=10, mFold=5):
        self.dataSet = dataSet
        self.linear = linear

        self.cv = cv

        self.mFold = mFold

        self.cs = [2**i for i in range(-5, 11)]
        self.gammas = [2**i for i in range(-15, 6)]



    def splitK(self,df,cv=10):
        df = df.sample(frac = 1)
        row = df.shape[0]
        train = int(row*(cv-1)/cv)
        traindata = df.iloc[:train,:]

        testdata = df.iloc[train:,:]

        return traindata,testdata

    def splittarget(self,df):
        data = df.iloc[:,1:]
        target = df.iloc[:,0]
        return data, target

    def splitm(self,df,m=5):
        df = df.sample(frac = 1)
        row = df.shape[0]
        train = int(row*(m-1)/m)
        trainData = df.iloc[:train,1:]
        trainTarget = df.iloc[:train,0]
        testData = df.iloc[train:,1:]
        testTarget= df.iloc[train:,0]


        return trainData,trainTarget,testData,testTarget





    def calculateClassMetrics(self, testClasses, prediction):


        cmat = confusion_matrix(testClasses, prediction)

        accuracy = cmat.diagonal() / cmat.sum(axis=1)
        score = precision_recall_fscore_support(testClasses, prediction)
        precision = score[0]
        recall = score[1]


        return accuracy,precision, recall


    def foldup(self):
        bestcs = []
        bestgs = [0.0] * 3

        for i in range(self.cv):
            train, test = self.splitK(self.dataSet)
            maxaccuracy = [0.0]*3
            bestgs = [0.0] * 3

            bestcs = [0.0] * 3
            for c in self.cs:
                for g in self.gammas:
                    accuracy = 0.0
                    for j in range(self.mFold):
                        traindata, traintarget, testdata, testtarget = self.splitm(train, self.mFold)

                        clf = OneVsRestClassifier(SVC(C=c, gamma=g))

                        clf.fit(traindata,traintarget)
                        prediction = clf.predict(testdata)
                        a,p,r = self.calculateClassMetrics(testtarget.T.values,prediction)
                        accuracy += a

                    accuracy /= self.mFold

                    for q in range(3):
                        if accuracy[q]>maxaccuracy[q]:
                            maxaccuracy[q] = accuracy[q]
                            bestcs[q] = c
                            bestgs[q] = g




            testd,testt = self.splittarget(test)
            traind,traint = self.splittarget(train)

            for c in range(3):
                clf = OneVsRestClassifier(SVC(C=bestcs[c],gamma=bestgs[c]))
                clf.fit(traind, traint)

                trainprediction = clf.predict(traind)

                ta,tp,tr = self.calculateClassMetrics(traint.T.values,trainprediction)




                prediction = clf.predict(testd)
                a,p,r = self.calculateClassMetrics(testt.T.values,prediction)

                print(i,c,bestcs[c], bestgs[c], ta[c], tp[c], tr[c], a[c], p[c], r[c])






def importData(fileLocation):
    dataSet = pd.read_csv(filepath_or_buffer=fileLocation, header=None)
    dataSet = dataSet.fillna(0)
    dataSet = dataSet.drop_duplicates()
    dataSet = shuffle(dataSet)

    return dataSet


wine = 'wine.csv'
wine = importData(wine)

warnings.filterwarnings('ignore')

MulticlassSVMRBF = SVMMultiRBF(wine)
MulticlassSVMRBF.foldup()

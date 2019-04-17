from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score

class SVMMulti:
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
        precision = cmat.diagonal()/cmat.sum(axis=1)
        recall = cmat.diagonal()/[(cmat[0][0]+cmat[1][0]+cmat[2][0]),(cmat[1][1]+cmat[0][1]+cmat[2][1]),(cmat[2][2]+cmat[1][2]+cmat[0][2])]

        return accuracy,precision, recall


    def foldup(self):
        bestcs = [0.0]*3
        for i in range(self.cv):
            train, test = self.splitK(self.dataSet)
            maxaccuracy = [0.0]*3

            bestcs = [0.0] * 3
            for c in self.cs:
                accuracy = 0.0
                for j in range(self.mFold):
                    traindata, traintarget, testdata, testtarget = self.splitm(train, self.mFold)

                    clf = OneVsRestClassifier(SVC(kernel='linear', C=c))

                    clf.fit(traindata,traintarget)
                    prediction = clf.predict(testdata)
                    a,p,r = self.calculateClassMetrics(testtarget.T.values,prediction)
                    accuracy += a

                accuracy /= self.mFold

                for q in range(3):
                    if accuracy[q]>maxaccuracy[q]:
                        maxaccuracy[q] = accuracy[q]
                        bestcs[q] = c



            testd,testt = self.splittarget(test)
            traind,traint = self.splittarget(train)

            for c in range(3):
                clf = OneVsRestClassifier(SVC(kernel='linear', C=bestcs[c]))
                clf.fit(traind, traint)

                trainprediction = clf.predict(traind)

                ta,tp,tr = self.calculateClassMetrics(traint.T.values,trainprediction)




                prediction = clf.predict(testd)
                a,p,r = self.calculateClassMetrics(testt.T.values,prediction)
                print(i,c,bestcs[c],ta[c],tp[c],tr[c],a[c],p[c],r[c])



from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle


class SVMMultiAUC:
    def __init__(self, dataSet, cv=10, m=5):
        self.dataSet = dataSet

        self.cv = cv

        self.m = m

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
        score = precision_recall_fscore_support(
            testClasses, prediction)
        precision = score[0]
        recall = score[1]

        return accuracy,precision, recall


    def foldup(self):
        bestcs = [0.0]*3
        fprss = []
        tprss = []
        for i in range(self.cv):
            print(i)

            train, test = self.splitK(self.dataSet)
            maxaccuracy = [0.0]*3


            for c in self.cs:
                accuracy = [0.0]*3
                for j in range(self.m):
                    traindata, traintarget, testdata, testtarget = self.splitm(train, self.m)
                    traintarget = traintarget.T

                    traintarget = label_binarize(traintarget, classes=[1, 2, 3])

                    testtarget = testtarget.T
                    testtarget = label_binarize(testtarget, classes=[1, 2, 3])
                    clf = OneVsRestClassifier(SVC(kernel='linear', C=c, probability=True))

                    clf.fit(traindata,traintarget)
                    predictionprob = clf.predict_proba(testdata)
                    for i in range(3):
                        fpr, tpr, threshpld = metrics.roc_curve(
                            testtarget[:, i], predictionprob[:, i])
                        accuracy[i] += metrics.auc(fpr, tpr)

                    for i in range(3):
                        if accuracy[i] >= maxaccuracy[i]:
                            maxaccuracy[i] = accuracy[i]
                            bestcs[i] = c



            testd,testt = self.splittarget(test)
            testt = testt.T
            testt = label_binarize(testt, classes=[1, 2, 3])
            traind,traint = self.splittarget(train)
            traint = traint.T
            traint = label_binarize(traint, classes=[1, 2, 3])
            fprs = []
            tprs = []
            for i in range(3):
                clf = OneVsRestClassifier(SVC(kernel='linear', C=bestcs[i], probability=True))
                clf.fit(traind, traint)



                predictionprob = clf.predict_proba(testd)
                fpr, tpr, thresholds = metrics.roc_curve(testt[:, i], predictionprob[:, i])
                print(fpr)
                print(tpr)
                print(metrics.auc(fpr, tpr))
                fprs.append(fpr)
                tprs.append(tpr)
            fprss.append(fprs)
            tprss.append(tprs)

        for c in range(3):

            print(c)
            plt.figure()
            for _ in range(self.cv):
                plt.plot(
                    fprss[_][c],
                    tprss[_][c],
                    lw=1,
                    label='Fold %d class = %d' % (_+1,c))
                plt.plot([0, 1], [0, 1], 'r--', lw=1)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.title(
                    'Receiver Operating Characteristic - Class {}'.format(
                        c + 1))
                plt.legend(loc='lower right')

                plt.savefig('SVMMulticlass_lin_ROC_{}'.format(c + 1))


def importData(fileLocation):
    dataSet = pd.read_csv(filepath_or_buffer=fileLocation, header=None)
    dataSet = dataSet.fillna(0)
    dataSet = dataSet.drop_duplicates()
    dataSet = shuffle(dataSet)

    return dataSet
wineFileLocation = 'wine.csv'
wineDataSet = importData(wineFileLocation)

wineMulticlassSVMAUC = SVMMultiAUC(wineDataSet,10,5)
wineMulticlassSVMAUC.foldup()
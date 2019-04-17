import math
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

class SVM:



    def importData(self,file):
        return np.array(pd.read_csv(file, header=None), dtype=np.float64)


    def normalizationTrain(self,train):
        mean = np.mean(train, 0)
        std = np.std(train, 0)
        data = (train - mean) / std
        data = self.addConstant(data)

        return data, mean, std


    def normalizationTest(self,test, mean, std):
        data = (test - mean) / std
        data = self.addConstant(data)

        return data


    def addConstant(self,train):
        return np.column_stack([np.ones([train.shape[0], 1]), train])
    
    
    def evaluate(self, trainLabel, prediction):


        precision = precision_score(trainLabel, prediction, average='weighted')
        recall = recall_score(trainLabel, prediction, average='weighted')

        accuracy = accuracy_score(trainLabel, prediction)

        return accuracy, precision, recall





    def mvalid(self,train, label, C, gamma, m=5):
        end = 0
        size = math.floor(label.size / m)
        accuracies = []

        for i in range(m):
            start = end
            end = start + size

            xval = train[start:end]
            yval = label[start:end]
            traindata = np.vstack([train[:start], train[end:]])
            traintarget = np.hstack([label[:start], label[end:]])

            clf = SVC(C=C, gamma=gamma)
            #        clf = SVC(C=C, kernel='linear')
            clf.fit(traindata, traintarget)

            accuracy = clf.score(xval, yval)
            accuracies.append(accuracy)

        return np.mean(accuracies)


    def validate(self,dataset, folds=10, m=5):
        np.random.shuffle(dataset)

        end = 0
        size = math.floor(dataset.shape[0] / folds)

        trainaccuracies = []
        trainprecisions = []
        trainrecalls = []

        testaccuracies = []
        testprecisions = []
        testrecalls = []

        gridc = []
        for i in [-5, -2, 0, 5, 10]:
            gridc.append(2 ** i)


        gridg = []

        for i in [-15, -10, -7,-5, 0]:
           gridg.append(2 ** i)


        for k in range(folds):
            start = end
            end = start + size
            testdataSet = dataset[start: end]

            left = dataset[0: start]
            right = dataset[end:]
            dataset_train = np.vstack([left, right])

            traindata = dataset_train[:, 0:-1]
            traintarget = dataset_train[:, -1]
            traindata, mean, std = self.normalizationTrain(traindata)

            testdata = testdataSet[:, 0:-1]
            testtarget = testdataSet[:, -1]
            testdata = self.normalizationTest(testdata, mean, std)

            maxaccuracy = -1
            for C in gridc:
                for gamma in gridg:
                    acc = self.mvalid(traindata, traintarget, C, gamma)
                    if acc > maxaccuracy:
                        maxaccuracy = acc
                        bestc = C
                        bestg = gamma

            print('max', maxaccuracy,' ',k,'C',bestc)
            print('g',bestg)


            #clf = SVC(C=bestc, gamma=bestg, probability=True)
            clf = SVC(C=bestc, kernel='linear', probability=True)
            clf.fit(traindata, traintarget)

            trainLabel = clf.predict(traindata)

            accuracy, precision, recall = self.evaluate(traintarget, trainLabel)

            trainaccuracies.append(accuracy)
            trainprecisions.append(precision)
            trainrecalls.append(recall)

            testLabel = clf.predict(testdata)

            accuracy, precision, recall = self.evaluate(testtarget, testLabel)

            testaccuracies.append(accuracy)
            testprecisions.append(precision)
            testrecalls.append(recall)

            prob = clf.predict_proba(testdata)
            f, t, thresholds = roc_curve(testtarget, prob[:, 1])
            rocauc = auc(f, t)
            plt.plot(f, t, lw=1, alpha=0.3, label= (k,rocauc))

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('false positive')
        plt.ylabel('true Positive')
        plt.legend(loc="upper right")
        plt.show()

        print('training data mean accuracy: ', np.mean(trainaccuracies),'Std accuracy: ', np.std(trainaccuracies),'mean precision: ', np.mean(trainprecisions),'Std precision: ', np.std(trainprecisions),'mean recall: ', np.mean(trainrecalls),'Std recall: ', np.std(trainrecalls))
        print('test data mean accuracy: ', np.mean(testaccuracies), 'Std accuracy: ', np.std(testaccuracies),
              'mean precision: ', np.mean(testprecisions), 'Std precision: ', np.std(testprecisions), 'mean recall: ',
              np.mean(testrecalls), 'Std recall: ', np.std(testrecalls))





    def foldup(self):
        print('bc')
        data = self.importData('breastcancer.csv')
        self.validate(data)
        print('diabetes')
        data = self.importData('diabetes.csv')
        self.validate(data)
        print('spam')
        data = self.importData('spambase.csv')
        self.validate(data)




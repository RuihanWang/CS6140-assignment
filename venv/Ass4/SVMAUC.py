import math
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

class SVMAUC:
    def importData(self, file):
        return np.array(pd.read_csv(file, header=None), dtype=np.float64)

    def normalizationTrain(self, data):
        mean = np.mean(data, 0)
        std = np.std(data, 0)
        datanew = (data - mean) / std
        datanew = self.addConstant(datanew)

        return datanew, mean, std

    def normalizationTest(self, data, mean, std):
        datanew = (data - mean) / std
        datanew = self.addConstant(datanew)

        return datanew

    def addConstant(self, data):
        return np.column_stack([np.ones([data.shape[0], 1]), data])

    def evaluate(self, trainLabel, prediction):

        precision = precision_score(trainLabel, prediction, average='weighted')
        recall = recall_score(trainLabel, prediction, average='weighted')

        accuracy = accuracy_score(trainLabel, prediction)

        return accuracy, precision, recall

    def mvalid(self,data, target, C, gamma, m=5):
        b = 0
        size = math.floor(target.size / m)
        aucs = []

        for i in range(m):
            a = b
            b = a + size

            testdata = data[a:b]
            testtarget = target[a:b]
            traindata = np.vstack([data[:a], data[b:]])
            traintarget = np.hstack([target[:a], target[b:]])

            clf = SVC(C=C, gamma=gamma, probability=True)
            #clf = SVC(C=C, kernel='linear', probability=True)
            clf.fit(traindata, traintarget)

            score = clf.predict_proba(testdata)[:, 1]
            p = roc_auc_score(testtarget, score)
            aucs.append(p)

        return np.mean(aucs)


    def validation(self,dataset, folds=10, m=5):
        np.random.shuffle(dataset)

        b = 0
        size = math.floor(dataset.shape[0] / folds)

        trainaccuracies = []
        trainprecisions = []
        trainrecalls = []

        testaccuracies = []
        testprecisions = []
        testrecalls = []
        cs = []
        for i in [-5, -2, 0, 5, 10]:
            cs.append(2 ** i)


        gs = []

        for i in [-15, -10, -7,-5, 0]:
           gs.append(2 ** i)

        for k in range(folds):
            a = b
            b = a + size
            dataset_test = dataset[a: b]

            left = dataset[0: a]
            right = dataset[b:]
            dataset_train = np.vstack([left, right])

            traindata = dataset_train[:, 0:-1]
            traintarget = dataset_train[:, -1]
            traindata, mean, std = self.normalizationTrain(traindata)

            testdata = dataset_test[:, 0:-1]
            testtarget = dataset_test[:, -1]
            testdata = self.normalizationTest(testdata, mean, std)

            maxauc = -1
            for C in cs:
                for gamma in gs:
                    a = self.mvalid(traindata, traintarget, C, gamma)
                    if a > maxauc:
                        maxauc = a
                        bestc = C
                        bestg = gamma

            print('max', maxauc,' ',k,'C',bestc,'g',bestg)

            clf = SVC(C=bestc, kernel='linear', probability=True)
            #clf = SVC(C=C, gamma=gamma, probability=True)
            clf.fit(traindata, traintarget)

            trainlabel = clf.predict(traindata)

            accuracy, precision, recall = self.evaluate(traintarget, trainlabel)

            trainaccuracies.append(accuracy)
            trainprecisions.append(precision)
            trainrecalls.append(recall)

            testlabel = clf.predict(testdata)

            accuracy, precision, recall = self.evaluate(testtarget, testlabel)

            testaccuracies.append(accuracy)
            testprecisions.append(precision)
            testrecalls.append(recall)

            prob= clf.predict_proba(testdata)

            fpr, tpr, thresholds = metrics.roc_curve(testtarget[:, c], proba[:, c])
            rocauc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label= (k,rocauc))


        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('false positive')
        plt.ylabel('true Positive')
        plt.legend(loc="upper right")
        plt.show()

        print('training data mean accuracy: ', np.mean(trainaccuracies), 'Std accuracy: ', np.std(trainaccuracies),
      'mean precision: ', np.mean(trainprecisions), 'Std precision: ', np.std(trainprecisions), 'mean recall: ',
      np.mean(trainrecalls), 'Std recall: ', np.std(trainrecalls))
        print('test data mean accuracy: ', np.mean(testaccuracies), 'Std accuracy: ', np.std(testaccuracies),
      'mean precision: ', np.mean(testprecisions), 'Std precision: ', np.std(testprecisions), 'mean recall: ',
      np.mean(testrecalls), 'Std recall: ', np.std(testrecalls))


    def foldup(self):
        data = self.importData('diabetes.csv')
        self.validation(data)


        data = self.importData('breastcancer.csv')
        self.validation(data)



        data = self.importData('spambase.csv')
        self.validation(data)



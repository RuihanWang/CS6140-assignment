import pandas as pd
import math
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score


class RegulatizedLogisticRegression:
    def __init__(self, data, filename,tolerance, a, lam=0.5,cv=10):

        self.filename = filename
        self.data = data
        self.tolerance = tolerance
        self.a = a
        self.lam = lam
        self.cv = cv

    def importData(self, fileLocation):
        dataSet = pd.read_csv(filepath_or_buffer=fileLocation, header=None)
        dataSet = dataSet.drop_duplicates()
        return dataSet

    def addConstant(self, data):
        data.insert(data.shape[1], data.shape[1], 1)
        return data

    def splitData(self, df, cv):
        df = df.sample(frac=1)
        row = df.shape[0]
        train = int(row * (cv - 1) / cv)
        trainData = df.iloc[:train, :(df.shape[1] - 1)]
        trainTarget = df.iloc[:train, (df.shape[1] - 1):]
        testData = df.iloc[train:, :df.shape[1] - 1]
        testTarget = df.iloc[train:, (df.shape[1] - 1):]

        return trainData, trainTarget, testData, testTarget

    def sigmoid(self, a):
        b = 1.0 / (np.exp(-a) + 1.0)
        return b

    def gradient(self, X, Y, theta):
        prediction = self.hypothesis(X, theta)
        error = prediction - Y

        return np.dot(X.T, error) / X.shape[0]

    def hypothesis(self, X, theta):
        return self.sigmoid(np.dot(X,theta.T))

    def predictError(self, data, target, w):



        probabilities = self.hypothesis(data, w)
        wnew = w.copy()
        wnew[0] = 0
        cost = (np.multiply(-target, np.log(probabilities)) - np.multiply(
            (1 - target), np.log(1 - probabilities))).mean() + (self.lam / (
                2 * data.shape[0])) * np.sum(np.square(wnew))

        return cost

    def grad(self, data, target, w):
        jresult = []
        count = 0
        pre = 0
        next = 0

        while self.converged(count, pre, next):
            pre = next
            next = self.predictError(data, target, w)

            g = self.sigmoid(data @ w.transpose())
            g.columns = [target.columns[0]]
            t = g - target
            gradient = self.gradient(data, target, w)
            wnew = w - (self.a * gradient).T



            count = count + 1
            w = wnew.copy()
            jresult.append(pre)

        return w, jresult

    def converged(self, count, pre, next):
        if count == 0: return True
        if count >= 1000: return False
        tolerance = float(self.tolerance)
        return math.fabs(pre - next) >= tolerance

    def normalizationTrain(self, data):
        mean = {}
        std = {}
        datanew = data.copy()

        for column in data:
            mean[column] = data[column].mean()
            std[column] = data[column].std()

        for index, row in data.iterrows():
            for label, content in row.iteritems():
                if std[column] == 0: continue
                nor = (content - mean[label]) / std[label]

                datanew.loc[index, label] = nor
        self.addConstant(datanew)
        return datanew, mean, std

    def normalizationTest(self, data, mean, std):

        datanew = data.copy()
        for index, row in data.iterrows():
            for label, content in row.iteritems():
                nor = (content - mean[label]) / std[label]

                datanew.loc[index, label] = nor

        self.addConstant(datanew)
        return datanew

    def predict(self, X, theta):
        probabilities = self.sigmoid(np.dot(X, theta.transpose()))

        return [
            1 if probability >= 0.5 else 0 for probability in probabilities
        ]

    def calculateClassMetrics(self, testClasses, prediction):

        accuracy = accuracy_score(testClasses, prediction)
        precision = precision_score(testClasses, prediction, average='weighted')
        recall = recall_score(testClasses, prediction, average='weighted')

        return accuracy, precision, recall

    def foldup(self):
        data = self.data
        tolerance = np.array([0.0005, 0.005, 0.05, ])
        learningRate = np.array([0.00005, 0.0005, 0.005])
        lam = np.array([0.05, 0.5, 5])

        dt = pd.read_csv(data, header=None)
        a = []
        p = []
        r = []
        jresult = []
        for i in range(self.cv):
            w = np.zeros((1, (dt.shape[1])))

            data, target, testd, testt = self.splitData(dt, 10)
            data, mean, std = self.normalizationTrain(data)
            testd = self.normalizationTest(testd, mean, std)
            w, jresult = self.grad(data, target, w)
            q = self.predict(testd, w)
            accuracy, precision, recall = self.calculateClassMetrics(testt, q)
            a.append(accuracy)
            p.append(precision)
            r.append(recall)
            print('finish', i)
        a.append(np.mean(a))
        a.append(np.std(a))
        p.append(np.mean(p))
        p.append(np.std(p))
        r.append(np.mean(r))
        r.append(np.std(r))

        result = pd.DataFrame({'Accuracy': a, 'Precision': p, 'recall': r})

        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        ax.table(cellText=result.values,
                 rowLabels=result.index,
                 colLabels=result.columns,
                 cellLoc='center', rowLoc='center',
                 loc='center')
        fig.tight_layout()
        plt.show()

        plt.plot(jresult)
        plt.xlabel('Iteration')
        plt.ylabel('LogisticLose')
        plt.title('Gradient Descent')
        plt.savefig('RegulatedRegression{}'.format(self.filename))



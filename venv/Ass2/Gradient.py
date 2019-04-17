import pandas as pd
import math
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


class Gradient:
    def __init__(self,data,tolerance,a,w,cv=10):
        self.data = data
        self.tolerance = tolerance
        self.a =a
        self.w = w
        self.cv = cv


    def importData(self,fileLocation):
        dataSet = pd.read_csv(filepath_or_buffer=fileLocation, header=None)
        dataSet = dataSet.drop_duplicates()
        return dataSet

    def splitData(self,df,cv):
        df = df.sample(frac = 1)
        row = df.shape[0]
        train = int(row*(cv-1)/cv)
        trainData = df.iloc[:train,:(df.shape[1]-1)]
        trainTarget = df.iloc[:train,(df.shape[1]-1):]
        testData = df.iloc[train:,:df.shape[1]-1]
        testTarget= df.iloc[train:,(df.shape[1]-1):]


        return trainData,trainTarget,testData,testTarget



    def predictError(self,data,target,w):
        p =data@w.transpose()

        target.columns = [p.columns[0]]

        err = p.add(target*(-1), fill_value=0)
        rms = math.sqrt(np.sum(err** 2)/data.shape[0])

        er = err.transpose()@data
        return rms,er


    def grad(self,data,target,w,a):
        jresult = []
        wnew = w.copy()
        count = 0
        pre = 0
        next = 0
        while self.converged(count,pre,next):
            pre = next
            next,er = self.predictError(data, target, w)
            for column in w:
                wnew[column] = w[column] - er[column]*a

            count = count+1
            w = wnew.copy()

            jresult.append(next)
        return w,jresult






    def converged(self,count,pre,next):
        if count == 0: return True
        if count >=1000: return False
        tolerance = float(self.tolerance)
        return  math.fabs(pre-next) >= tolerance

    def normalizationTrain(self,data):
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

        datanew[datanew.shape[1]] = 1
        return datanew,mean,std

    def normalizationTest(self,data,mean,std):

        datanew = data.copy()
        for index, row in data.iterrows():
            for label,content in row.iteritems():
                nor = (content - mean[label]) / std[label]

                datanew.loc[index, label] = nor

        datanew[datanew.shape[1]] = 1
        return datanew
    def validate(self):
        data,target,testd,testt = self.splitData(self.data,self.cv)
        dt,mean, std = self.normalizationTrain(data)

        dataTest = self.normalizationTest(testd,mean,std)

        wmin,jresult = self.grad(dt,target,self.w,self.a)

        testrms = self.predict(dataTest,testt,wmin)

        return testrms,jresult,wmin


    def predict(self,data,target,wmin):
        rms,er = self.predictError(data,target,wmin)
        return rms



    def foldup(self):
        result = []
        test = []
        jre=[]
        for i in range(self.cv):
            testrms,jresult,wmin = self.validate()
            result.append(jresult[-1])
            test.append(testrms)
            jre = jresult
        plt.plot(jre)
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.title('Gradient Descent')
        plt.show()
        return np.mean(result),np.std(result),np.mean(test),np.std(test),result,test







from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator

class MultiNomial:
    def __init__(self):
        self.priorProbability = {}
        self.conditionalProbability = {}

    def computeData(self, trainData,trainlabel):
        testData = open(trainData)

        df = pd.read_csv(testData, delimiter=' ', names=['docIdx', 'wordIdx', 'count'])
        docIdx = df['docIdx'].values

        trainLabel = open(trainlabel)
        label = []
        lines = trainLabel.readlines()
        for line in lines:
            label.append(int(line.split()[0]))
        i = 0
        nLabel = []
        for index in range(len(docIdx) - 1):
            nLabel.append(label[i])
            if docIdx[index] != docIdx[index + 1]:
                i += 1
        nLabel.append(label[i])
        df['classIdx'] = nLabel
        # PriorProbility
        trainLabel = open(trainlabel)
        p = pd.read_csv(trainLabel, delimiter=' ', names=['classIdx'])
        t = p.groupby('classIdx').size().sum()
        PriorP = (p.groupby('classIdx').size()) / t
        #words
        words = df.groupby('wordIdx').sum().sort_values('count', ascending=False)['count']
        return df,PriorP,words

    def selectVoc(self,words,n):

        # select vocabalury

        vocabulary = words.loc[:n]
        return vocabulary

    def train(self, df, totalWords):
        # Total words need to be adjusted to vocabalury
        pwordclass = df.groupby(['classIdx', 'wordIdx'])
        pclass = df.groupby(['classIdx'])
        PrMN = (pwordclass['count'].sum() + 1) / (pclass['count'].sum() + totalWords)
        PrMN = PrMN.unstack()
        for c in range(1, 21):
            PrMN.loc[c, :] = PrMN.loc[c, :].fillna(1 / (pclass['count'].sum()[c] + totalWords))
        PrMN = np.log(PrMN)
        PrMNDict = PrMN.to_dict()
        return PrMNDict

    def test(self, testdata,Pr_dict,pi,words):
        testData = open(testdata)

        df = pd.read_csv(testData, delimiter=' ', names=['docIdx', 'wordIdx', 'count'])

        # Using dictionaries for greater speed
        dfDict = df.to_dict()
        newDict = {}
        prediction = []

        # newDict = {docIdx : {wordIdx: count},....}
        for idx in range(len(dfDict['docIdx'])):
            docIdx = dfDict['docIdx'][idx]
            wordIdx = dfDict['wordIdx'][idx]
            count = dfDict['count'][idx]
            try:
                newDict[docIdx][wordIdx] = count
            except:
                newDict[dfDict['docIdx'][idx]] = {}
                newDict[docIdx][wordIdx] = count

        # Calculating the scores for each doc
        for docIdx in range(1, len(newDict) + 1):
            scoreDict = {}
            # Creating a probability row for each class
            for classIdx in range(1, 21):
                scoreDict[classIdx] = 0
                # For each word:
                for wordIdx in newDict[docIdx]:
                    if(wordIdx in words.index):

                        try:
                            probability = Pr_dict[wordIdx][classIdx]
                            power = np.log(1 + newDict[docIdx][wordIdx])

                            scoreDict[classIdx] += power * probability
                        except:
                            scoreDict[classIdx] += 0
                            # f*log(Pr(i|j))

                scoreDict[classIdx] += np.log(pi[classIdx])

                # Get class with max probabilty for the given docIdx
            max_score = max(scoreDict, key=scoreDict.get)
            prediction.append(max_score)

        return prediction

    def calculateClassMetrics(self, testLabel, prediction):

        testClasses = [int(s) for s in open(testLabel).read().split()]

        precision = precision_score(testClasses, prediction, average='weighted')
        recall = recall_score(testClasses, prediction, average='weighted')
        accuracy = accuracy_score(testClasses, prediction)

        return precision,recall,accuracy
    def calculateClassesMetrics(self, testLabel, prediction):

        testClasses = [int(s) for s in open(testLabel).read().split()]

        precision = precision_score(testClasses, prediction, average=None)
        recall = recall_score(testClasses, prediction, average=None)
        accuracy = precision_score(testClasses, prediction, average=None)

        return precision,recall,accuracy

    def foldup(self, trainData, trainLabel, testData, testLabel):

        a=[]
        p=[]
        r=[]
        df,priorp,words = self.computeData(trainData,trainLabel)
        vocabularySize = [100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 25000, 50000, words.shape[0]]
        print("Size\tAccuracy\tPrecision\tRecall")
        for size in vocabularySize:
            voc = self.selectVoc(words,size)
            pr = self.train(df,size)
            prediction = self.test(testData,pr,priorp,voc)
            pr, re, ac = self.calculateClassMetrics(testLabel,prediction)
            print(pr)
            a.append(ac)
            p.append(pr)
            r.append(re)
            if size == words.shape[0]:
                pp,rr,aa = self.calculateClassesMetrics(testLabel, prediction)


        return a,p,r,words.shape[0],pp,rr,aa




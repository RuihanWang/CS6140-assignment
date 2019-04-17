import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score
from PerceptronDual import PerceptronDual
from Perceptron import Perceptron
from PerceptronKernel import PerceptronKernel
from RegulatizedLogisticRegression import RegulatizedLogisticRegression
from SVMAUC import SVMAUC
from SVM import SVM
from SVMMulti import SVMMulti
from SVMMultiRBF import SVMMultiRBF
from SVMMultiAUCRBF import SVMMultiAUCRBF


def importData(fileLocation):
    data = pd.read_csv(filepath_or_buffer=fileLocation, header=None,dtype=float64)
    data = data.fillna(0)
    data = data.drop_duplicates()
    data = shuffle(data)

    return data









perceptronDataFileLocation = 'perceptronData.csv'
d = importData(perceptronDataFileLocation)
p = Perceptron(d)
print(p.foldup())


d = importData(perceptronDataFileLocation)
p = PerceptronDual(d)
print(p.foldup())

a = 'twoSpirals.csv'
a = importData(a)



d = importData(a)
p = PerceptronDual(d)
print(p.foldup())




b = PerceptronKernel(a)
b.foldup()







p = RegulatizedLogisticRegression('diabetes.csv','diabetes',0.000001,0.05,5)
p.foldup()

p = RegulatizedLogisticRegression('spambase.csv','spambase',0.0005, 0.5,5)
p.foldup()


p = RegulatizedLogisticRegression('breastcancer.csv','bc',0.0001,0.1)
p.foldup()


s = SVM()
s.foldup()


s = SVMAUC()
s.foldup()





wine = 'wine.csv'
wine = importData(wine)


MulticlassSVM = SVMMulti(wine)
MulticlassSVM.foldup()


MulticlassSVMRBF = SVMMultiRBF(wine)
MulticlassSVMRBF.foldup()



MulticlassSVMAUC = SVMMultiAUC(wine)
MulticlassSVMAUC.foldup()



MulticlassSVMAUCRBF = SVMMultiRBF(wine)
MulticlassSVMAUCRBF.foldup()
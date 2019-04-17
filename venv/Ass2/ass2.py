import pandas as pd
import math
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from Gradient import Gradient
from NormalEquation import NormalEquation
from polyNomial import poly
from ridge import ridge


#1 Gradient Descent for 3 datasets
a=[0.0004,0.001,0.0007]
tolerance = [0.005,0.001,0.0001]


data = pd.read_csv('housing.csv',header = None)
dt = pd.DataFrame(data = data)
w =  pd.DataFrame(data = np.zeros((1,(dt.shape[1]))))
pp = Gradient(dt,tolerance[0],a[0],w)
print(pp.foldup())


data = pd.read_csv('yachtData.csv',header = None)
dt = pd.DataFrame(data = data)
w =  pd.DataFrame(data = np.zeros((1,(dt.shape[1]))))
pp = Gradient(dt,tolerance[1],a[1],w)
print(pp.foldup())






data = pd.read_csv('concreteData.csv',header = None)
dt = pd.DataFrame(data = data)
w =  pd.DataFrame(data = np.zeros((1,(dt.shape[1]))))
pp = Gradient(dt,tolerance[2],a[2],w)
print(pp.foldup())


#2 Normal Equation for 3 dataset
data = pd.read_csv('housing.csv',header = None)
dt = pd.DataFrame(data = data)
dataSet = dt.drop_duplicates()
mal = NormalEquation()
print(mal.foldup(dataSet,10))




data = pd.read_csv('yachtData.csv',header = None)
dt = pd.DataFrame(data = data)
dataSet = dt.drop_duplicates()
mal = NormalEquation()
print(mal.foldup(dataSet,10))



data = pd.read_csv('concreteData.csv',header = None)
dt = pd.DataFrame(data = data)
dataSet = dt.drop_duplicates()
mal = NormalEquation()
print(mal.foldup(dataSet,10))




#3 polynomial test for yacht and sinious datasets
data = pd.read_csv('yachtData.csv',header = None)
dt = pd.DataFrame(data = data)
dataSet = dt.drop_duplicates()
mal = poly(7)
mal.foldup(dataSet,10,7)


data = pd.read_csv('sinData_Train.csv',header = None)
dt2 = pd.DataFrame(data = data)
data2 = dt2.drop_duplicates()
valid = pd.read_csv('sinData_Validation.csv',header = None)
valid2 = pd.DataFrame(data = valid)
valid2 = valid2.drop_duplicates()
mal = poly(15)
mal.foldu(data2,valid2,15)


#4 Ridge regression

data = pd.read_csv('sinData_Train.csv',header = None)
dt = pd.DataFrame(data = data)
rr = ridge(5,10)
print(rr.foldup(dt))
rr = ridge(9,10)
print(rr.foldup(dt))

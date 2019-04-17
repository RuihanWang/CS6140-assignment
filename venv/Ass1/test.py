import pandas as pd
import math
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

d = {
    'Name': ['Alisa', 'Bobby', 'jodha', 'jack', 'raghu', 'Cathrine',
             'Alisa', 'Bobby', 'kumar', 'Alisa', 'Alex', 'Cathrine'],
    'Age': [26, 24, 23, 22, 23, 24, 26, 24, 22, 23, 24, 24],

    'Score': [5, 6, 0, 0, 0, 0, 18, 0, 20, 0, 0, 0]}


predict = [15.382352941176471, 15.382352941176471, 11.750704225352113, 11.750704225352113, 11.750704225352113, 15.382352941176471, 15.382352941176471, 20.838805970149252, 20.838805970149252, 20.838805970149252, 18.91904761904762, 15.382352941176471, 15.382352941176471, 11.750704225352113, 11.750704225352113, 15.382352941176471, 20.838805970149252, 20.838805970149252, 37.69041095890411, 11.750704225352113, 11.750704225352113, 15.382352941176471, 11.750704225352113, 11.750704225352113, 20.838805970149252, 20.838805970149252, 27.699999999999992, 37.69041095890411, 18.91904761904762, 20.838805970149252, 20.838805970149252, 15.382352941176471, 20.838805970149252, 19.106666666666666, 19.106666666666666, 19.106666666666666, 19.106666666666666, 20.838805970149252, 18.91904761904762, 20.838805970149252, 19.106666666666666, 19.106666666666666, 18.91904761904762, 20.838805970149252, 19.106666666666666, 20.838805970149252, 27.699999999999992, 21.406666666666673, 37.69041095890411, 27.699999999999992, 21.406666666666673]
realresult =  [14.1,
 12.7,
 13.5,
 14.9,
 20,
 16.4,
 17.7,
 19.5,
 20.2,
 21.4,
 19.9,
 19.0,
 19.1,
 19.1,
 20.1,
 19.9,
19.6,
 23.2,
 29.8,
 13.8,
 13.3,
 16.7,
 12. ,
 14.6,
 21.4,
 23. ,
 23.7,
 25. ,
 21.8,
 20.6,
 21.2,
 19.1,
 20.6,
 15.2,
  7.0 ,
  8.1,
 13.6,
 20.1,
 21.8,
 24.5,
 23.1,
 19.7,
 18.3,
 21.2,
 17.5,
 16.8,
 22.4,
 20.6,
 23.9,
 22.0,
 11.9]






df = pd.DataFrame(d, columns=['Name', 'Age', 'Score'])
items = df.shape[0]
def entropy():
    c = df.columns[-1]
    d = df[c].value_counts()
    d = d/items
    entro = 0
    print(d)
    for e in d:
        entro -= e*np.log2(e)
    return entro
def SD(dt):

    target= dt.iloc[:,(dt.shape[1]-1):]
    sd = np.sum((target-target.mean())**2)


    return sd

def BinarysplitData(data, attribute, value, ):
    split = []
    try:
        var = int(value)

        data1 = data[data[attribute] >= value]
        data2 = data[data[attribute] <= value]
        split.append(data1)
        split.append(data2)
    except:
        data1 = data[data[attribute] == value]
        data2 = data[data[attribute] != value]
        split.append(data1)
        split.append(data2)
    return split

def sumOfSquaredErrors(dt):

    target = dt.iloc[:, (dt.shape[1] - 1):]

    sd = np.sum((target - target.mean()) ** 2)
    return sd




def normalizationTrain(data):

    for column in data:
        max = np.amax(data[column])
        min = np.amin(data[column])
        data[column] = (data[column]- min) / (max - min)

    return data

def BinarysplitData(data, attribute, value):
    split = []
    data1 = data[data[attribute] >= value]
    data2 = data[data[attribute] < value]
    split.append(data1)
    split.append(data2)
    print(split)
    return split

def getSDgain(data, attribute, value):
    sd = SD(data)
    split = BinarysplitData(data, attribute, value)
    sdnew = 0.0
    for dt in split:
        sdnew += SD(dt)

    gain = sd - sdnew;
    print(gain)
def isPure(data):
    return data.iloc[:,-1].unique().shape[0] == 1


def MultisplitData(data,attribute):
    split = []
    uniquevalue = data[attribute].unique()

    for v in uniquevalue:
        datanew = data[data[attribute] == v]

        split.append(datanew)

    return split


attributes = df.columns
score = df['Score'].values
age = df['Age'].values



print(MultisplitData(df,'Age'))

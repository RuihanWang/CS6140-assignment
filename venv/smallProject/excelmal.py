import pandas as pd
import math
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import re


def tsplit(string, delimiters):
    """Behaves str.split but supports multiple delimiters."""

    delimiters = tuple(delimiters)
    stack = [string, ]

    for delimiter in delimiters:
        for i, substring in enumerate(stack):
            substack = substring.split(delimiter)
            stack.pop(i)
            for j, _substring in enumerate(substack):
                stack.insert(i + j, _substring)

    return stack


data = pd.read_excel("test.xls")
dt = pd.DataFrame(data = data)
possibleaddress = (dt.iloc[:,[8]])


temp  = possibleaddress[possibleaddress['买家留言'].str.contains("省")==True]


for index,row in temp.iterrows():
    print(tsplit(row['买家留言'],('省','市','区')))


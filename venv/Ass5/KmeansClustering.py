import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

class KmeansClustering:
    def __init__(self, K, file_location):
        self.data = self.importData(file_location)
        self.K = K


    def importData(self, file_location):
        data = pd.read_csv(filepath_or_buffer=file_location, header=None,dtype=float64)
        data = data.fillna(0)
        data = data.drop_duplicates()

        return data

    def SSE(self, data, K_centers, Z_n):
        Z_n@K_centers.T




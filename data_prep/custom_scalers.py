import numpy as np
from sklearn.preprocessing import StandardScaler

def load_scaler_function(scaling_type):
    print("Scaling type: ", scaling_type)
    if scaling_type == "standard":
        return StandardScaler()

    elif scaling_type == "custom":
        return CustomScaler()
    else:
        raise

class CustomScaler(object):

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, data):
        self.scaler.fit(data)

    def transform(self, data):
        trans = self.scaler.transform(data)
        trans[:, 34] = data[:, 34]/100. # Age
        trans[:, 35] = data[:, 35] # Gender
        trans[:, 36] = data[:, 36] # Unit1
        trans[:, 37] = data[:, 37] # Unit2
        trans[:, 39] = data[:, 39]/350. # ICULOS
        return trans

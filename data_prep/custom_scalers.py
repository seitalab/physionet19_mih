import numpy as np
from sklearn.preprocessing import StandardScaler

def load_scaler_function(scaling_type):
    print("Scaling type: ", scaling_type)
    if scaling_type == "standard":
        return StandardScaler()

    elif scaling_type == "custom_v1":
        return CustomScalerV1()
    elif scaling_type == "custom_v2":
        return CustomScalerV2()
    elif scaling_type == "custom_v3":
        return CustomScalerV3()
    elif scaling_type == "d_only":
        return CustomScalerV4()
    elif scaling_type == "concat_v1":
        return CustomScalerV5()
    else:
        raise

class CustomScalerV1(object):

    def __init__(self):
        self.scaler_m = StandardScaler()
        self.scaler_f = StandardScaler()

    def separate_by_gender(self, data):
        gender = data[:, 35]
        data_m = data[gender==1]
        data_f = data[gender==0]
        return data_m, data_f

    def is_male_data(self, data):
        return data[:,35][0] == 1

    def fit(self, data):
        data_m, data_f = self.separate_by_gender(data)
        self.scaler_m.fit(data_m)
        self.scaler_f.fit(data_f)
        
    def transform(self, data):
        if self.is_male_data(data):
            return self.scaler_m.transform(data)
        else: return self.scaler_f.transform(data)

class CustomScalerV2(CustomScalerV1):

    age1 = 50
    age2 = 70
    
    def __init__(self):
        self.scaler_m1 = StandardScaler()
        self.scaler_m2 = StandardScaler()
        self.scaler_m3 = StandardScaler()
        self.scaler_f1 = StandardScaler()
        self.scaler_f2 = StandardScaler()
        self.scaler_f3 = StandardScaler()

    def separate_by_age(self, data):
        age = data[:, 34]
        data_1 = data[age < self.age1]
        data_2 = data[(age >= self.age1) & (age < self.age2)]
        data_3 = data[age >= self.age2]
        return data_1, data_2, data_3
    
    def get_age_category(self, data):
        age = data[:, 34][0]
        if age < self.age1: return "category1"
        elif age >= self.age2: return "category3"
        else: return "category2"
        
    def fit(self, data):
        data_m, data_f = self.separate_by_gender(data)
        data_m1, data_m2, data_m3 = self.separate_by_age(data_m)
        data_f1, data_f2, data_f3 = self.separate_by_age(data_f)
        self.scaler_m1.fit(data_m1)
        self.scaler_m2.fit(data_m2)
        self.scaler_m3.fit(data_m3)
        self.scaler_f1.fit(data_f1)
        self.scaler_f2.fit(data_f2)
        self.scaler_f3.fit(data_f3)

    def transform(self, data):
        category = self.get_age_category(data)
        if category == "category1":
            if self.is_male_data(data):
                return self.scaler_m1.transform(data)
            else: return self.scaler_f1.transform(data)
        elif category == "category2":
            if self.is_male_data(data):
                return self.scaler_m2.transform(data)
            else: return self.scaler_f2.transform(data)
        else:
            if self.is_male_data(data):
                return self.scaler_m3.transform(data)
            else: return self.scaler_f3.transform(data)

class CustomScalerV3(object):

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

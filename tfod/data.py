
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_subseq(ts, look_back, pred_length):
        sub_seq, next_values = [], []
        for i in range(len(ts)-look_back-pred_length):  
            sub_seq.append(ts[i:i+look_back])
            next_values.append(ts[i+look_back:i+look_back+pred_length].T[0])
        return sub_seq, next_values


def load(name='ecg', split=0.2):
    if name.lower() == "ecg":
        df = pd.read_csv("http://www.cs.ucr.edu/~eamonn/discords/qtdbsel102.txt", header=None, delimiter='\t')
        ecg = df.iloc[:,2].values
        ecg = ecg.reshape(len(ecg), -1)
        print('length of ECG data : ', len(ecg))

        # standardize
        scaler = StandardScaler()
        std_ecg = scaler.fit_transform(ecg)
        std_ecg = std_ecg[:5000]

        look_back = 10
        pred_length = 3

        sub_seq, next_values = create_subseq(std_ecg, look_back, pred_length)   
        return np.array(sub_seq), np.array(next_values), std_ecg
    return

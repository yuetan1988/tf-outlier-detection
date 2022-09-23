
import os
import pandas as pd 
import numpy as np 

from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_ecg(base_dir):
    df = pd.read_csv(os.path.join(base_dir, 'ECG_data/qtdbsel102.txt'), header=None, delimiter='\t')
    ecg = df.iloc[:,2].values
    ecg = ecg.reshape(len(ecg), -1)
    print('length of ECG data : ', len(ecg))

    # standardize
    scaler = StandardScaler()
    std_ecg = scaler.fit_transform(ecg)
    std_ecg = std_ecg[5000:]
    return std_ecg


def load_test_ecg(base_dir):
    def create_subseq(ts, look_back, pred_length):
        sub_seq, next_values = [], []
        for i in range(len(ts)-look_back-pred_length):  
            sub_seq.append(ts[i:i+look_back])
            next_values.append(ts[i+look_back:i+look_back+pred_length].T[0])
        return sub_seq, next_values

    df = pd.read_csv(os.path.join(base_dir, 'ECG_data/qtdbsel102.txt'), header=None, delimiter='\t')
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
    # X_train, X_test, y_train, y_test = train_test_split(sub_seq, next_values, test_size=0.2)

    # X_train = np.array(X_train)
    # X_test = np.array(X_test)
    # y_train = np.array(y_train)
    # y_test = np.array(y_test)
    return np.array(sub_seq), np.array(next_values), std_ecg


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    base_dir = './data'
    std_ecg = load_ecg(base_dir)    

    plt.style.use('ggplot')
    plt.figure(figsize=(15,5))
    plt.xlabel('time')
    plt.ylabel('ECG\'s value')
    plt.plot(np.arange(5000), std_ecg[:5000], color='b')
    plt.ylim(-3, 3)
    x = np.arange(4200,4400)
    y1 = [-3]*len(x)
    y2 = [3]*len(x)
    plt.fill_between(x, y1, y2, facecolor='g', alpha=.3)
    plt.savefig('p.png')

    normal_cycle = std_ecg[5000:]
    plt.figure(figsize=(10,5))
    plt.title("training data")
    plt.xlabel('time')
    plt.ylabel('ECG\'s value')
    plt.plot(np.arange(5000,8000), normal_cycle[:3000], color='b')# stop plot at 8000 times for friendly visual
    plt.savefig('p2.png')

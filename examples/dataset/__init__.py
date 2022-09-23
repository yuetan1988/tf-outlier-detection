
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.read_ecg import load_ecg, load_test_ecg
from dataset.load_data import DataReader, DataLoader


def load_data(name, base_dir, only_test=False):
    if name.lower() == 'ecg':
        if only_test:
            x_test, y_test, sig = load_test_ecg(base_dir)
            return x_test, y_test, sig
        else:
            data = load_ecg(base_dir)
            
            train, valid = train_test_split(data, test_size=0.2)    
            return train, valid


import sys
sys.path.append('..')

import numpy as np
import functools
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input

from dataset import load_data, DataReader, DataLoader
from tfod import TsModel, OdModel, KerasTrainer


def build_model(model):
    inputs = Input([10, 1])
    outputs = model(inputs)
    return tf.keras.Model(inputs, outputs)


def run_lstm():
    train, valid= load_data('ecg', base_dir='./data')
    train_data = DataReader(train)
    valid_data = DataReader(valid)
    print(len(train_data), len(valid_data))
    print(train_data[20][0].shape, train_data[20][1].shape)

    # plt.plot(range(10), train_data[20][0])
    # plt.plot(range(10, 13), train_data[20][1])
    # plt.savefig('p.png')
    # assert 0==1

    
    train_loader = DataLoader(train_data)(batch_size=128)
    valid_loader = DataLoader(valid_data)(batch_size=128)

    # for x, y in train_loader:
    #    print(x.shape, y.shape)   
    # TODO: 以上传数据的方式是有问题的


    # x_test, y_test, sig = load_data('ecg', base_dir='./data', only_test=True)   

    # backbone = TsModel(use_model='rnn')
    # ts_model = functools.partial(build_model, backbone)

    # trainer = KerasTrainer(ts_model)
    # trainer.train((x_test, y_test), (x_test, y_test), n_epochs=50)
    # trainer.save_model(model_dir='./weights/lstm.h5')

    
    x_test, y_test, sig = load_data('ecg', base_dir='./data', only_test=True)
    backbone = build_model(TsModel(use_model='rnn'))
    backbone.load_weights('./weights/lstm.h5')
 
    detect_model = OdModel(backbone, train_sequence_length=10)
    det = detect_model.detect(x_test, y_test)
    detect_model.plot(sig, det)


if __name__ == '__main__':
    run_lstm()

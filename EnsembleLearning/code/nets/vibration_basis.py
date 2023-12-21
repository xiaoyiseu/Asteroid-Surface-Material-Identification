from __future__ import print_function
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Embedding
from tensorflow.python.keras.layers import LSTM, SimpleRNN, GRU
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from tensorflow.python.keras import callbacks
import tensorflow as tf
import torch
import random
import pickle


def create_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation = "softmax", kernel_regularizer = tf.keras.regularizers.l2(0.000001))])
    model.compile(
    optimizer = tf.keras.optimizers.SGD(lr = 5),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ['sparse_categorical_accuracy'])
    return model

def vibration_detect(csv_path):
    '''导入待测文件'''
    data = pd.read_csv(csv_path, header=None)
    C = data.iloc[:,4000]
    T = data.iloc[:,1:4000]

    scaler = Normalizer().fit(T)
    testT = scaler.transform(T)

    y_test = np.array(C)
    x_test = np.reshape(testT, (testT.shape[0], 1, testT.shape[1]))



    new_model = create_model()
    new_model.build((x_test.shape))
    new_model.load_weights(r"D:\Trans\Experiment 5 signal_transfer_image experiment\Ensemble_learning\model_data\weights_vibration\grulayer_model_final2.hdf5")


    y_pred = []
    y_true = []
    for i in range(len(x_test)):
        x_test1 = x_test[i]
        X_test = tf.reshape(x_test1, (1, 1, 3999))
        y_pred.append(int(new_model.predict_classes(X_test)))
        # y_pred = list(new_model.predict_classes(X_test))


    list2 = "\n".join(str(i) for i in y_pred)
    # d = len(y_pred)
    f = r"D:\Trans\Experiment 5 signal_transfer_image experiment\Ensemble_learning\datasets\results\vibration.txt"

    with open(f, "a") as file:
        # for i in range(d):
        file.write(list2 + " " + "\n")


if __name__ == "__main__":
    path = r'D:\Trans\Experiment 5 signal_transfer_image experiment\Ensemble_learning\datasets\digit\test1.csv'
    vibration_detect(path)
    print("振动数据检测完毕!")
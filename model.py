
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc  # 计算roc和auc
import os


def q_dataset(dataset, look_back):
    # 这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


def q_data(_data, num):
    _datacy = pd.concat([analysis_data[(_data['failure'] == 1)].sample(
        num), _data[(_data['failure'] == 0)].sample(num)], axis=0)
    return _datacy


#def smote_data(_data, num):
#    return _datacy

def q_selection(X_sele, Y_sele):
    fs = feature_selection.SelectPercentile(
        feature_selection.chi2, percentile=60)
    X_re = fs.fit_transform(X_sele, Y_sele)
    return X_re


def q_learn(x_train_, y_train_):
    # model = MultinomialNB(alpha = 0.1)
    # model.fit(x_train_, y_train_)
    model = MultinomialNB().fit(x_train_, y_train_)
    return model

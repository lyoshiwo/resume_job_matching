# encoding=utf8
import numpy as np
from sklearn import cross_validation
import pandas as pd
import os
import time
from keras.models import Sequential, model_from_json
import util
from pandas import Series


def score_lists(list_1, list_2):
    count = 0
    total = len(list_1)
    print total
    for i in range(total):
        if list_1[i] == list_2[i]:
            count += 1
    return float(count) / total


def evaluate_k_recall(k, y_test, y_proba_list):
    count = 0
    for i in range(len(y_test)):
        s = Series(y_proba_list[i])
        s.sort_values(inplace=True)
        pre_k = s.index.values[-1 * k:]
        if y_test[i] in pre_k:
            count += 1
    print float(count) / (len(y_test))


print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def get_ensemble_score(name):
    if os.path.exists(util.features_prefix + name + "_XXXYYY.pkl") is False:
        print 'file does not exist'
        exit()
    [X_train, X_validate, X_test, y_train, y_validate, y_test] = pd.read_pickle(
        util.features_prefix + name + '_XXXYYY.pkl')
    import xgboost as xgb

    rf_clf_2 = pd.read_pickle(util.models_prefix + name + '_rf.pkl')
    list_all = []
    rf_2_list = rf_clf_2.predict_proba(X_test)
    from sklearn.feature_selection import SelectFromModel
    list_all.append(rf_2_list)
    xgb_2 = xgb.Booster({'nthread': 4})  # init model
    xgb_2.load_model(util.models_prefix + name + '_xgb_prob.pkl')  # load data
    dtest = xgb.DMatrix(X_test)
    xgb_2_test = xgb_2.predict(dtest)
    list_all.append(xgb_2_test)
    # list_all.append(xgb_1_test)
    import copy
    [train_X, train_Y] = pd.read_pickle(util.features_prefix + name + '_XY.pkl')
    X_semantic = np.array(copy.deepcopy(X_test[:, range(95, 475)]))
    X_manual = np.array(copy.deepcopy(X_test[:, range(0, 95)]))
    X_cluster = np.array(copy.deepcopy(X_test[:, range(475, 545)]))
    X_document = np.array(copy.deepcopy(X_test[:, range(545, 547)]))
    X_document[:, [0]] = X_document[:, [0]] + train_X[:, [-1]].max()
    X_semantic = X_semantic.reshape(X_semantic.shape[0], 10, -1)
    X_semantic_1 = np.zeros((X_semantic.shape[0], X_semantic.shape[2], X_semantic.shape[1]))
    for i in range(int(X_semantic.shape[0])):
        X_semantic_1[i] = np.transpose(X_semantic[i])
    json_string = pd.read_pickle(util.models_prefix + name + '_json_string_cnn.pkl')
    model_cnn = model_from_json(json_string)
    model_cnn.load_weights(util.models_prefix + name + '_nn_weight_cnn.h5')
    cnn_list = model_cnn.predict_proba([X_document, X_cluster, X_manual, X_semantic_1])
    # cnn_list_prob = model_cnn.predict_proba([X_document, X_cluster, X_manual, X_semantic_1])
    kk = list(cnn_list)
    list_all.append(kk)
    json_string = pd.read_pickle(util.models_prefix + name + '_json_string_lstm.pkl')
    model_lstm = model_from_json(json_string)
    model_lstm.load_weights(util.models_prefix + name + '_nn_weight_lstm.h5')
    lstm_list = model_lstm.predict_proba([X_document, X_cluster, X_manual, X_semantic_1])
    # cnn_list_prob = model_cnn.predict_proba([X_document, X_cluster, X_manual, X_semantic_1])
    kk = list(lstm_list)
    list_all.append(kk)
    temp_list = []
    for i in range(len(y_test)):
        temp = np.zeros(len(list_all[0][0]))
        for z in list_all:
            temp += np.array(z[i])
        temp_list.append(temp)
    evaluate_k_recall(1, y_test, temp_list)

    print '**************************'


if __name__ == "__main__":
    for name in ['degree', 'position', 'salary', 'size']:
        get_ensemble_score(name)
        # xg
        # 2016 - 07 - 16
        # 23:39:28
        # 2016 - 07 - 16
        # 23:58:37
        # 2016 - 07 - 17
        # 00:34:06

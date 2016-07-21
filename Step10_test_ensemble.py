# encoding=utf8
import numpy as np
from sklearn import cross_validation
import pandas as pd
import os
import time
from keras.models import Sequential, model_from_json
import util


def score_lists(list_1, list_2):
    count = 0
    total = len(list_1)
    print total
    for i in range(total):
        if list_1[i] == list_2[i]:
            count += 1
    return float(count) / total


print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def get_esembel_score(name):
    if os.path.exists(util.features_prefix + name + "_XXXYYY.pkl") is False:
        print 'file does not exist'
        exit()
    [X_train, X_validate, X_test, y_train, y_validate, y_test] = pd.read_pickle(
        util.features_prefix + name + '_XXXYYY.pkl')
    import xgboost as xgb

    rf_clf_2 = pd.read_pickle(util.models_prefix + name+'_rf.pkl')
    list_all = []
    rf_2_list = rf_clf_2.predict(X_test)
    from sklearn.feature_selection import SelectFromModel

    model = SelectFromModel(rf_clf_2, prefit=True)
    temp = model.get_support()
    print sum(temp)
    list_all.append(rf_2_list)
    print rf_clf_2.score(X_test, y_test)
    xgb_2 = xgb.Booster({'nthread': 4})  # init model
    xgb_2.load_model(util.models_prefix +name+ '_xgb.pkl')  # load data
    print len(xgb_2.get_fscore().keys())
    dtest = xgb.DMatrix(X_test)
    xgb_2_test = xgb_2.predict(dtest)
    list_all.append(xgb_2_test)
    print score_lists(xgb_2_test, y_test)
    from keras.utils import np_utils
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
    json_string = pd.read_pickle(util.models_prefix +name+ '_json_string_cnn.pkl')
    model_cnn = model_from_json(json_string)
    model_cnn.load_weights(util.models_prefix + name+'_nn_weight_cnn.h5')
    cnn_list = model_cnn.predict_classes([X_document, X_cluster, X_manual, X_semantic_1])
    # cnn_list_prob = model_cnn.predict_proba([X_document, X_cluster, X_manual, X_semantic_1])
    kk = list(cnn_list)
    list_all.append(kk)
    print score_lists(kk, y_test)
    json_string = pd.read_pickle(util.models_prefix + name + '_json_string_lstm.pkl')
    model_lstm = model_from_json(json_string)
    model_lstm.load_weights(util.models_prefix + name + '_nn_weight_lstm.h5')
    lstm_list = model_lstm.predict_classes([X_document, X_cluster, X_manual, X_semantic_1])
    # cnn_list_prob = model_cnn.predict_proba([X_document, X_cluster, X_manual, X_semantic_1])
    kk = list(lstm_list)
    list_all.append(kk)
    print score_lists(kk, y_test)
    list_ensemble = []
    for i in range(len(y_test)):
        dict_all = {}
        for z in range(len(list_all)):
            dict_all[list_all[z][i]] = dict_all.setdefault(list_all[z][i], 0) + 1
            tmp_list = dict_all.items()
        list_ensemble.append(sorted(tmp_list, lambda a, b: -cmp(a[1], b[1]))[0][0])
    print score_lists(list_ensemble, y_test)
    print '**************************'


if __name__ == "__main__":
    for name in ['degree', 'position', 'salary', 'size']:
        get_esembel_score(name)
        # xg
        # 2016 - 07 - 16
        # 23:39:28
        # 2016 - 07 - 16
        # 23:58:37
        # 2016 - 07 - 17
        # 00:34:06

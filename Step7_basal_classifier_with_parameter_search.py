# encoding=utf8
from sklearn import cross_validation, grid_search
import pandas as pd
import os
import time
import util
from matplotlib import pyplot as plt

print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

tree_test = False
xgb_test = False
cnn_test = False
rf_test = False
lstm_test = False
# max_depth=12 0.433092948718

# rn>200,rh=21 0.537 to 0.541; rh=21,rn=400; 0.542


CV_FLAG = 1
param = {}
param['objective'] = 'multi:softmax'
param['eta'] = 0.03
param['max_depth'] = 6
param['eval_metric'] = 'merror'
param['silent'] = 1
param['min_child_weight'] = 10
param['subsample'] = 0.7
param['colsample_bytree'] = 0.2
param['nthread'] = 4
param['num_class'] = -1


def get_all_by_name(name):
    import numpy as np
    if os.path.exists(util.features_prefix + name + "_XY.pkl") is False:
        print name + 'file does not exist'
        exit()
    if os.path.exists(util.features_prefix + name + '_XXXYYY.pkl') is False:
        [train_X, train_Y] = pd.read_pickle(util.features_prefix + name + '_XY.pkl')
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_X, train_Y, test_size=0.33,
                                                                             random_state=0)
        X_train, X_validate, y_train, y_validate = cross_validation.train_test_split(X_train, y_train, test_size=0.33,
                                                                                     random_state=0)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        X_test = np.array(X_test)
        X_validate = np.array(X_validate)
        y_validate = np.array(y_validate)
        pd.to_pickle([X_train, X_validate, X_test, y_train, y_validate, y_test],
                     util.features_prefix + name + '_XXXYYY.pkl')
    if os.path.exists(util.features_prefix + name + '_XXXYYY.pkl'):
        print name
        from sklearn.ensemble import RandomForestClassifier
        if rf_test is False:
            [train_X, train_Y] = pd.read_pickle(util.features_prefix + name + '_XY.pkl')
            [X_train, X_validate, X_test, y_train, y_validate, y_test] = pd.read_pickle(
                util.features_prefix + name + '_XXXYYY.pkl')
            x = np.concatenate([X_train, X_validate], axis=0)
            y = np.concatenate([y_train, y_validate], axis=0)
            print 'rf'
            n_estimator = range(100, 301, 100)
            max_depth = range(5, 26, 1)
            clf = RandomForestClassifier(n_jobs=4)
            parameters = {'n_estimators': n_estimator, 'max_depth': max_depth}
            grid_clf = grid_search.GridSearchCV(clf, parameters)
            grid_clf.fit(np.array(train_X), np.array(train_Y))
            score = grid_clf.grid_scores_
            l1 = [1 - x[1] for x in score if x[0]['n_estimators'] == n_estimator[0]]
            l2 = [1 - x[1] for x in score if x[0]['n_estimators'] == n_estimator[1]]
            l3 = [1 - x[1] for x in score if x[0]['n_estimators'] == n_estimator[2]]
            plt.plot(range(5, 26, 1), l1,
                     'b--')
            plt.plot(range(5, 26, 1), l2,
                     'r.--')
            plt.plot(range(5, 26, 1), l3,
                     'g')
            plt.legend((str(n_estimator[0]) + ' estimators', str(n_estimator[1]) + ' estimators',
                        str(n_estimator[2]) + ' estimators'),
                       loc=0, shadow=True)
            plt.xlabel('max depth of RandomForest')
            plt.ylabel('average error rate of  3-fold cross-validation')
            plt.grid(True)
            plt.show()
            exit()
            print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        if xgb_test is False:
            [train_X, train_Y] = pd.read_pickle(util.features_prefix + name + '_XY.pkl')
            print 'xg'
            import xgboost as xgb

            print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            set_y = set(train_Y)
            param["num_class"] = len(set_y)
            dtrain = xgb.DMatrix(train_X, label=train_Y)
            xgb.cv(param, dtrain, 4, nfold=3, show_progress=True)

        if cnn_test is False:
            [train_X, train_Y] = pd.read_pickle(util.features_prefix + name + '_XY.pkl')
            print 'cnn'
            import copy
            import numpy as np
            from sklearn.preprocessing import LabelEncoder
            from keras.utils import np_utils
            from keras.layers.convolutional import Convolution1D

            label_dict = LabelEncoder().fit(train_Y)
            label_num = len(label_dict.classes_)
            x = train_X
            y = train_Y
            train_Y = np_utils.to_categorical(y, label_num)
            # x = np.concatenate([X_train, X_validate], axis=0)
            X_train = x
            X_semantic = np.array(copy.deepcopy(X_train[:, range(95, 475)]))
            X_manual = np.array(copy.deepcopy(X_train[:, range(0, 95)]))
            X_cluster = np.array(copy.deepcopy(X_train[:, range(475, 545)]))
            X_document = np.array(copy.deepcopy(X_train[:, range(545, 547)]))
            X_document[:, [0]] = X_document[:, [0]] + train_X[:, [-1]].max()
            dic_num_cluster = X_cluster.max()
            dic_num_manual = train_X.max()
            dic_num_document = X_document[:, [0]].max()
            from keras.models import Sequential
            from keras.layers.embeddings import Embedding
            from keras.layers.core import Merge
            from keras.layers.core import Dense, Dropout, Activation, Flatten
            from keras.layers.recurrent import LSTM

            X_semantic = X_semantic.reshape(X_semantic.shape[0], 10, -1)
            X_semantic_1 = np.zeros((X_semantic.shape[0], X_semantic.shape[2], X_semantic.shape[1]))
            for i in range(int(X_semantic.shape[0])):
                X_semantic_1[i] = np.transpose(X_semantic[i])
            model_semantic = Sequential()
            model_lstm = Sequential()
            model_lstm.add(LSTM(output_dim=30, input_shape=X_semantic_1.shape[1:], go_backwards=True))
            model_semantic.add(Convolution1D(nb_filter=32,
                                             filter_length=2,
                                             border_mode='valid',
                                             activation='relu', input_shape=X_semantic_1.shape[1:]))
            # model_semantic.add(MaxPooling1D(pool_length=2))
            model_semantic.add(Convolution1D(nb_filter=8,
                                             filter_length=2,
                                             border_mode='valid',
                                             activation='relu'))
            # model_semantic.add(MaxPooling1D(pool_length=2))
            model_semantic.add(Flatten())

            # we use standard max pooling (halving the output of the previous layer):
            model_manual = Sequential()
            model_manual.add(Embedding(input_dim=dic_num_manual + 1, output_dim=20, input_length=X_manual.shape[1]))
            # model_manual.add(Convolution1D(nb_filter=2,
            #                                filter_length=2,
            #                                border_mode='valid',
            #                                activation='relu'))
            # model_manual.add(MaxPooling1D(pool_length=2))
            # model_manual.add(Convolution1D(nb_filter=8,
            #                                filter_length=2,
            #                                border_mode='valid',
            #                                activation='relu'))
            # model_manual.add(MaxPooling1D(pool_length=2))
            model_manual.add(Flatten())

            model_document = Sequential()
            model_document.add(
                Embedding(input_dim=dic_num_document + 1, output_dim=2, input_length=X_document.shape[1]))
            model_document.add(Flatten())

            model_cluster = Sequential()
            model_cluster.add(Embedding(input_dim=dic_num_cluster + 1, output_dim=5, input_length=X_cluster.shape[1]))
            model_cluster.add(Flatten())
            model = Sequential()
            # model = model_cluster
            model.add(Merge([model_document, model_cluster, model_manual, model_semantic], mode='concat',
                            concat_axis=1))
            model.add(Dense(512))
            model.add(Dropout(0.5))
            model.add(Activation('relu'))
            model.add(Dense(128))
            model.add(Dropout(0.5))
            model.add(Activation('relu'))
            # We project onto a single unit output layer, and squash it with a sigmoid:
            model.add(Dense(label_num))
            model.add(Activation('softmax'))

            model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                          metrics=['accuracy'])
            # model.fit(X_cluster_1, train_Ymetrics=['accuracy'], batch_size=100,
            #           nb_epoch=100, validation_split=0.33, verbose=1)
            model.fit([X_document, X_cluster, X_manual, X_semantic_1], train_Y,
                      batch_size=100, nb_epoch=15, validation_split=0.33)
            print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if lstm_test is False:
            import numpy as np

            [train_X, train_Y] = pd.read_pickle(util.features_prefix + name + '_XY.pkl')
            print 'lstm'
            import copy
            import numpy as np
            from sklearn.preprocessing import LabelEncoder
            from keras.utils import np_utils
            from keras.layers.convolutional import Convolution1D

            label_dict = LabelEncoder().fit(train_Y)
            label_num = len(label_dict.classes_)
            train_Y = np_utils.to_categorical(train_Y, label_num)
            # x = np.concatenate([X_train, X_validate], axis=0)
            X_train = train_X
            X_semantic = np.array(copy.deepcopy(X_train[:, range(95, 475)]))
            X_manual = np.array(copy.deepcopy(X_train[:, range(0, 95)]))
            X_cluster = np.array(copy.deepcopy(X_train[:, range(475, 545)]))
            X_document = np.array(copy.deepcopy(X_train[:, range(545, 547)]))
            X_document[:, [0]] = X_document[:, [0]] + train_X[:, [-1]].max()
            dic_num_cluster = X_cluster.max()
            dic_num_manual = train_X.max()
            dic_num_document = X_document[:, [0]].max()
            from keras.models import Sequential
            from keras.layers.embeddings import Embedding
            from keras.layers.core import Merge
            from keras.layers.core import Dense, Dropout, Activation, Flatten
            from keras.layers.recurrent import LSTM

            X_semantic = X_semantic.reshape(X_semantic.shape[0], 10, -1)
            X_semantic_1 = np.zeros((X_semantic.shape[0], X_semantic.shape[2], X_semantic.shape[1]))
            for i in range(int(X_semantic.shape[0])):
                X_semantic_1[i] = np.transpose(X_semantic[i])
            model_semantic = Sequential()
            model_lstm = Sequential()
            model_lstm.add(LSTM(output_dim=30, input_shape=X_semantic_1.shape[1:], go_backwards=True))
            model_semantic.add(Convolution1D(nb_filter=32,
                                             filter_length=2,
                                             border_mode='valid',
                                             activation='relu', input_shape=X_semantic_1.shape[1:]))
            # model_semantic.add(MaxPooling1D(pool_length=2))
            model_semantic.add(Convolution1D(nb_filter=8,
                                             filter_length=2,
                                             border_mode='valid',
                                             activation='relu'))
            # model_semantic.add(MaxPooling1D(pool_length=2))
            model_semantic.add(Flatten())

            # we use standard max pooling (halving the output of the previous layer):
            model_manual = Sequential()
            model_manual.add(Embedding(input_dim=dic_num_manual + 1, output_dim=20, input_length=X_manual.shape[1]))
            model_manual.add(Flatten())

            model_document = Sequential()
            model_document.add(
                Embedding(input_dim=dic_num_document + 1, output_dim=2, input_length=X_document.shape[1]))
            model_document.add(Flatten())

            model_cluster = Sequential()
            model_cluster.add(Embedding(input_dim=dic_num_cluster + 1, output_dim=5, input_length=X_cluster.shape[1]))
            model_cluster.add(Flatten())
            model = Sequential()
            # model = model_cluster
            model.add(Merge([model_document, model_cluster, model_manual, model_lstm], mode='concat',
                            concat_axis=1))
            model.add(Dense(512))
            model.add(Dropout(0.5))
            model.add(Activation('relu'))
            model.add(Dense(128))
            model.add(Dropout(0.5))
            model.add(Activation('relu'))
            # We project onto a single unit output layer, and squash it with a sigmoid:
            model.add(Dense(label_num))
            model.add(Activation('softmax'))

            model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
            # model.fit(X_cluster_1, train_Y, batch_size=100,
            #           nb_epoch=100, validation_split=0.33, verbose=1)
            model.fit([X_document, X_cluster, X_manual, X_semantic_1], train_Y,
                      batch_size=100, nb_epoch=15, validation_split=0.33)
            print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

if __name__ == "__main__":
    for name in ['degree', 'salary', 'size', 'position']:
        get_all_by_name(name)

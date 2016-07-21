# encoding=utf8
from sklearn import cross_validation
import pandas as pd
import os
import time
from keras.preprocessing import sequence
import util

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
keys = {'salary': [353, 2, 7], 'size': [223, 3, 6], 'degree': [450, 4, 8], 'position': [390, 7, 16]}

# keys = {'salary': [1,1,1], 'size': [1,1,1], 'degree': [1,1,1], 'position': [1,1,1]}


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
        from sklearn.ensemble import RandomForestClassifier

        if rf_test is False:
            [train_X, train_Y] = pd.read_pickle(util.features_prefix + name + '_XY.pkl')
            [X_train, X_validate, X_test, y_train, y_validate, y_test] = pd.read_pickle(
                util.features_prefix + name + '_XXXYYY.pkl')
            x = np.concatenate([X_train, X_validate], axis=0)
            y = np.concatenate([y_train, y_validate], axis=0)
            print 'rf'
            print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # for max in range(12, 23, 5):
            clf = RandomForestClassifier(n_jobs=4, n_estimators=400, max_depth=22)
            clf.fit(x, y)
            print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            pd.to_pickle(clf, util.models_prefix + name + '_rf.pkl')
            y_p = clf.predict(X_test)
            print name + ' score:' + util.score_lists(y_test, y_p)

        if xgb_test is False:
            [train_X, train_Y] = pd.read_pickle(util.features_prefix + name + '_XY.pkl')
            [X_train, X_validate, X_test, y_train, y_validate, y_test] = pd.read_pickle(
                util.features_prefix + name + '_XXXYYY.pkl')
            x = np.concatenate([X_train, X_validate], axis=0)
            y = np.concatenate([y_train, y_validate], axis=0)
            print 'xg'
            import xgboost as xgb

            print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            set_y = set(train_Y)
            param["num_class"] = len(set_y)

            x = np.concatenate([X_train, X_validate], axis=0)
            y = np.concatenate([y_train, y_validate], axis=0)
            dtrain = xgb.DMatrix(x, label=y)
            param['objective'] = 'multi:softmax'
            xgb_2 = xgb.train(param, dtrain, keys[name][0])
            print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            xgb_2.save_model(util.models_prefix + name + '_xgb.pkl')
            dtest = xgb.DMatrix(X_test)
            y_p = xgb_2.predict(dtest)
            print name + ' score:' + util.score_lists(y_test, y_p)
            param['objective'] = 'multi:softprob'
            dtrain = xgb.DMatrix(x, label=y)
            xgb_1 = xgb.train(param, dtrain, keys[name][0])
            xgb_1.save_model(util.models_prefix + name + '_xgb_prob.pkl')

        if cnn_test is False:
            [train_X, train_Y] = pd.read_pickle(util.features_prefix + name + '_XY.pkl')
            [X_train, X_validate, X_test, y_train, y_validate, y_test] = pd.read_pickle(
                util.features_prefix + name + '_XXXYYY.pkl')
            print 'cnn'
            import copy
            import numpy as np
            from sklearn.preprocessing import LabelEncoder
            from keras.utils import np_utils
            from keras.layers.convolutional import Convolution1D
            print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            label_dict = LabelEncoder().fit(train_Y)
            label_num = len(label_dict.classes_)
            x = np.concatenate([X_train, X_validate], axis=0)
            y = np.concatenate([y_train, y_validate], axis=0)
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

            model.compile(loss='categorical_crossentropy', optimizer='adadelta')
            # model.fit(X_cluster_1, train_Y, batch_size=100,
            #           nb_epoch=100, validation_split=0.33, verbose=1)
            model.fit([X_document, X_cluster, X_manual, X_semantic_1], train_Y,
                      batch_size=100, nb_epoch=keys[name][1])
            print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            json_string = model.to_json()
            pd.to_pickle(json_string, util.models_prefix + name + '_json_string_cnn.pkl')
            model.save_weights(util.models_prefix + name + '_nn_weight_cnn.h5')
            X_semantic = np.array(copy.deepcopy(X_test[:, range(95, 475)]))
            X_manual = np.array(copy.deepcopy(X_test[:, range(0, 95)]))
            X_cluster = np.array(copy.deepcopy(X_test[:, range(475, 545)]))
            X_document = np.array(copy.deepcopy(X_test[:, range(545, 547)]))
            X_document[:, [0]] = X_document[:, [0]] + train_X[:, [-1]].max()
            X_semantic = X_semantic.reshape(X_semantic.shape[0], 10, -1)
            X_semantic_1 = np.zeros((X_semantic.shape[0], X_semantic.shape[2], X_semantic.shape[1]))
            for i in range(int(X_semantic.shape[0])):
                X_semantic_1[i] = np.transpose(X_semantic[i])
            cnn_list = model.predict_classes([X_document, X_cluster, X_manual, X_semantic_1])
            print name + ' score:' + util.score_lists(y_test, cnn_list)

        if lstm_test is False:
            import numpy as np

            [train_X, train_Y] = pd.read_pickle(util.features_prefix + name + '_XY.pkl')
            [X_train, X_validate, X_test, y_train, y_validate, y_test] = pd.read_pickle(
                util.features_prefix + name + '_XXXYYY.pkl')
            x = np.concatenate([X_train, X_validate], axis=0)
            y = np.concatenate([y_train, y_validate], axis=0)
            print 'lstm'
            import copy
            import numpy as np
            from sklearn.preprocessing import LabelEncoder
            from keras.utils import np_utils
            from keras.layers.convolutional import Convolution1D

            label_dict = LabelEncoder().fit(train_Y)
            label_num = len(label_dict.classes_)
            x = np.concatenate([X_train, X_validate], axis=0)
            y = np.concatenate([y_train, y_validate], axis=0)
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

            model.compile(loss='categorical_crossentropy', optimizer='adadelta')
            # model.fit(X_cluster_1, train_Y, batch_size=100,
            #           nb_epoch=100, validation_split=0.33, verbose=1)
            model.fit([X_document, X_cluster, X_manual, X_semantic_1], train_Y,
                      batch_size=100, nb_epoch=keys[name][2])
            print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            json_string = model.to_json()
            pd.to_pickle(json_string, util.models_prefix + name + '_json_string_lstm.pkl')
            model.save_weights(util.models_prefix + name + '_nn_weight_lstm.h5')
            X_semantic = np.array(copy.deepcopy(X_test[:, range(95, 475)]))
            X_manual = np.array(copy.deepcopy(X_test[:, range(0, 95)]))
            X_cluster = np.array(copy.deepcopy(X_test[:, range(475, 545)]))
            X_document = np.array(copy.deepcopy(X_test[:, range(545, 547)]))
            X_document[:, [0]] = X_document[:, [0]] + train_X[:, [-1]].max()
            X_semantic = X_semantic.reshape(X_semantic.shape[0], 10, -1)
            X_semantic_1 = np.zeros((X_semantic.shape[0], X_semantic.shape[2], X_semantic.shape[1]))
            for i in range(int(X_semantic.shape[0])):
                X_semantic_1[i] = np.transpose(X_semantic[i])
            lstm_list = model.predict_classes([X_document, X_cluster, X_manual, X_semantic_1])
            print name + ' score:' + util.score_lists(y_test, lstm_list)


if __name__ == "__main__":
    for name in ['salary', 'size', 'degree', 'position']:
        print name
        get_all_by_name(name)

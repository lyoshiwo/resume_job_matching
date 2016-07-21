# encoding=utf8
from gensim.models import Word2Vec
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import sys
from sklearn_pandas import DataFrameMapper
import util
import pandas as pd

reload(sys)
sys.setdefaultencoding('utf-8')

features = \
    [
        "id", "major", "age", "gender",
        "isenglish", "isjunior", "isbachelor", "ismaster", "isintern",
        "total_previous_job",
        "last_salary", "last_size", "last_position_name", "last_industry", "last_type", "last_type1", "last_department",
        "last_start_year", "last_start_month", "last_end_year", "last_end_month", "last_interval_month",
        "third_salary", "third_size", "third_position_name", "third_industry", "third_type", "third_type1",
        "third_department",
        "third_start_year", "third_start_month", "third_end_year", "third_end_month", "third_interval_month",
        "first_salary", "first_size", "first_position_name", "first_industry", "first_type", "first_type1",
        "first_department",
        "first_start_year", "first_start_month", "first_end_year", "first_end_month", "first_interval_month",
        "last3_interval_month", "diff_last3_size", "diff_last3_salary", "diff_last3_industry",
        "diff_last3_position_name",
        "total_interval_month", "diff_salary", "diff_size", "diff_industry", "diff_position_name",
        "major_1",
        "last_position_name_1", "last_department_1",
        "third_position_name_1", "third_department_1",
        "first_position_name_1", "first_department_1",
        "major_2",
        "last_position_name_2", "last_department_2",
        "third_position_name_2", "third_department_2",
        "first_position_name_2", "first_department_2",
        "start_working_age", "rev_working_age", "pre_working_month", "pre_interval_month",
         "pre_largest_size", "pre_largest_salary",
        "pre_least_size",
        "pre_least_salary",
        "pre_size1",
        "pre_size2",
        "pre_size3",
        "pre_size4",
        "pre_size5",
        "pre_size6",
        "pre_size7",
        "pre_salary1",
        "pre_salary2",
        "pre_salary3",
        "pre_salary4",
        "pre_salary5",
        "pre_salary6",
        "pre_salary7",

        "promotion_size",
        "promotion_salary",
        "decrease_size",
        "decrease_salar"
    ]
all_features = features + ["predict_degree", "predict_salary", "predict_size", "predict_position_name"]

train = pd.read_pickle(util.features_prefix + "manual_feature.pkl")
print len(train), len(features), len(all_features)
train = train[all_features]
train = train[train["predict_position_name"].isin(util.position_name_list)]
data_all = pd.concat([train[features]])


def get_mapper(data_all):
    param_list = [
        ('id', None),
        ('major', LabelEncoder()),
        ('age', None),
        ('gender', LabelEncoder()),
        ('isenglish', None),
        ('isjunior', None),
        ('isbachelor', None),
        ('ismaster', None),
        ('isintern', None),
        ('total_previous_job', None),
        ('last_type', LabelEncoder()),
        ('last_type1', LabelEncoder()),
        ('last_department', LabelEncoder()),
        ('last_size', None),
        ('last_salary', None),
        ('last_industry', LabelEncoder()),
        ('last_position_name', LabelEncoder()),
        ('last_start_year', None),
        ('last_start_month', None),
        ('last_end_year', None),
        ('last_end_month', None),
        ('last_interval_month', None),
        ('third_type', LabelEncoder()),
        ('third_type1', LabelEncoder()),
        ('third_department', LabelEncoder()),
        ('third_size', None),
        ('third_salary', None),
        ('third_industry', LabelEncoder()),
        ('third_position_name', LabelEncoder()),
        ('third_start_year', None),
        ('third_start_month', None),
        ('third_end_year', None),
        ('third_end_month', None),
        ('third_interval_month', None),
        ('first_type', LabelEncoder()),
        ('first_type1', LabelEncoder()),
        ('first_department', LabelEncoder()),
        ('first_size', None),
        ('first_salary', None),
        ('first_industry', LabelEncoder()),
        ('first_position_name', LabelEncoder()),
        ('first_start_year', None),
        ('first_start_month', None),
        ('first_end_year', None),
        ('first_end_month', None),
        ('first_interval_month', None),
        ('last3_interval_month', None),
        ('diff_last3_salary', LabelEncoder()),
        ('diff_last3_size', LabelEncoder()),
        ('diff_last3_industry', LabelEncoder()),
        ('diff_last3_position_name', LabelEncoder()),
        ('total_interval_month', None),
        ('diff_salary', LabelEncoder()),
        ('diff_size', LabelEncoder()),
        ('diff_industry', LabelEncoder()),
        ('diff_position_name', LabelEncoder()),
        ('major_1', LabelEncoder()),
        ('last_position_name_1', LabelEncoder()),
        ('last_department_1', LabelEncoder()),
        ('third_position_name_1', LabelEncoder()),
        ('third_department_1', LabelEncoder()),
        ('first_position_name_1', LabelEncoder()),
        ('first_department_1', LabelEncoder()),
        ('major_2', LabelEncoder()),
        ('last_position_name_2', LabelEncoder()),
        ('last_department_2', LabelEncoder()),
        ('third_position_name_2', LabelEncoder()),
        ('third_department_2', LabelEncoder()),
        ('first_position_name_2', LabelEncoder()),
        ('first_department_2', LabelEncoder()),
        ('start_working_age', None),
        ('rev_working_age', None),
        ('pre_working_month', None),
        ('pre_interval_month', None),
        ("pre_largest_size", None),
        ("pre_largest_salary", None),
        ("pre_least_size", None),
        ("pre_least_salary", None),
        ("pre_size1", None),
        ("pre_size2", None),
        ("pre_size3", None),
        ("pre_size4", None),
        ("pre_size5", None),
        ("pre_size6", None),
        ("pre_size7", None),
        ("pre_salary1", None),
        ("pre_salary2", None),
        ("pre_salary3", None),
        ("pre_salary4", None),
        ("pre_salary5", None),
        ("pre_salary6", None),
        ("pre_salary7", None),

        ("promotion_size", None),
        ("promotion_salary", None),
        ("decrease_size", None),
        ("decrease_salar", None)
    ]
    print "the mapper's param list is %s" % (len(param_list))
    mapper = DataFrameMapper(param_list)
    mapper.fit(data_all)
    return mapper


mapper = get_mapper(data_all)


def getPrecision(multiclf, train_X, train_Y, label_dict):
    pred_Y = multiclf.predict(train_X)
    pred_Y = [int(p) for p in pred_Y]

    print "total accuracy_score%s" % (accuracy_score(train_Y, pred_Y))
    diff_num = len(label_dict.classes_)

    for i in xrange(diff_num):
        hit, test_cnt, pred_cnt = 0, 0, 0
        for k in xrange(len(train_Y)):
            if train_Y[k] == i:
                test_cnt += 1
            if pred_Y[k] == i:
                pred_cnt += 1
            if train_Y[k] == i and pred_Y[k] == i:
                hit += 1
        print "\t\t%s %d %d %d\tprecision_score %s\trecall_score %s" % (
            label_dict.inverse_transform([i])[0], hit, test_cnt, pred_cnt, hit * 1.0 / (pred_cnt + 0.01),
            hit * 1.0 / (test_cnt + 0.01))


def get_feature_by_experienceList(workExperienceList, c_k_64_dic):
    level_two = [u'industry', u'department', u'type', u'position_name']
    feature_list = []
    for k in [0, -1]:
        for i in level_two:
            try:
                feature_list.append(c_k_64_dic[workExperienceList[k][i]])
            except Exception, e:
                feature_list.append(-1)
    return feature_list


level_one = [u'major', u'degree', u'gender', u'age', u'workExperienceList', u'_id', u'id']
level_two = [u'salary', u'end_date', u'industry', u'position_name', u'department', u'type', u'start_date', u'size']


def sentence_to_matrix_vec(sentence, model, featuresNum, k_mean_dict_1, k_mean_dict_2):
    temp = np.zeros((featuresNum * (7 * 5 + 3) + 7 * 5 * 2))
    if sentence == None: return temp

    num = (len(sentence) - 3) / 7 if (len(sentence) - 3) / 7 <= 5 else 5
    for i in range(num * 7):
        temp[featuresNum * i:featuresNum * (i + 1)] = model[sentence[i]]
        try:
            temp[38 * featuresNum + num * 2] = k_mean_dict_1[sentence[i]]
            temp[38 * featuresNum + num * 2 + 1] = k_mean_dict_2[sentence[i]]
        except Exception, e:
            continue
    for i in range(3):
        temp[(5 * 7 + i) * featuresNum:(5 * 7 + i + 1) * featuresNum] = model[sentence[-1 * (i + 1)]]
    return temp


def getAllFeatures(train, mapper):
    print "this is getAllFeatures"
    # every record has a cluster value calculated by lda
    w2c_f, w2c_w = 10, 14
    lda_dict_1 = util.read_dict(util.features_prefix + 'id_lda_256.pkl')
    lda_dict_2 = util.read_dict(util.features_prefix + 'id_lda_512.pkl')
    k_mean_dict_1 = util.read_dict(util.features_prefix + 'c_k_all_64.pkl')
    k_mean_dict_2 = util.read_dict(util.features_prefix + 'c_k_all_128.pkl')
    sentence_dict_path = util.txt_prefix + 'id_sentences.pkl'
    word2vec_path = util.txt_prefix + str(w2c_f) + 'features_1minwords_' + str(w2c_w) + 'context.pkl'
    sentence_dic = util.read_dict(sentence_dict_path)
    model = Word2Vec.load(word2vec_path)

    train_X = train[features]
    train_X = mapper.transform(train_X)  # .values
    new_train_X = []
    for i in xrange(len(train_X)):
        id = train_X[i][0]
        lda_1 = lda_dict_1[id]
        lda_2 = lda_dict_2[id]
        s = sentence_dic.get(id)
        f = np.concatenate(([train_X[i][1:].astype(np.float32)],
                            [sentence_to_matrix_vec(s, model, w2c_f, k_mean_dict_1, k_mean_dict_2)]), axis=1)[0]
        f = np.concatenate(([f], [[lda_1, lda_2]]), axis=1)[0]
        new_train_X.append(f)
    new_train_X = np.array(new_train_X)
    return new_train_X


if __name__ == "__main__":
    train_Y = []
    train_X = []
    test_X = []
    import os

    train_X = getAllFeatures(train, mapper)
    if os.path.exists(util.features_prefix + "/position_XY.pkl") is False:
        train_Y = list(train["predict_position_name"].values)
        label_dict = LabelEncoder().fit(train_Y)
        label_dict_classes = len(label_dict.classes_)
        train_Y = label_dict.transform(train_Y)
        pd.to_pickle([train_X, train_Y], util.features_prefix + "/position_XY.pkl")
    else:
        [train_X, train_Y] = pd.read_pickle(util.features_prefix + "/position_XY.pkl")
        print len(train_X[0]), len(train_Y)
        print 95 + 380 + 7 * 5 * 2 + 2
        print train_X[0]

    if os.path.exists(util.features_prefix + "/degree_XY.pkl") is False:
        train_Y = list(train["predict_degree"].values)
        label_dict = LabelEncoder().fit(train_Y)
        label_dict_classes = len(label_dict.classes_)
        train_Y = label_dict.transform(train_Y)
        pd.to_pickle([train_X, train_Y], util.features_prefix + "/degree_XY.pkl")
    else:
        [train_X, train_Y] = pd.read_pickle(util.features_prefix + "/degree_XY.pkl")
        print len(train_X[0]), len(train_Y)

    if os.path.exists(util.features_prefix + "/size_XY.pkl") is False:
        train_Y = list(train["predict_size"].values)
        label_dict = LabelEncoder().fit(train_Y)
        label_dict_classes = len(label_dict.classes_)
        train_Y = label_dict.transform(train_Y)
        pd.to_pickle([train_X, train_Y], util.features_prefix + "/size_XY.pkl")
    else:
        [train_X, train_Y] = pd.read_pickle(util.features_prefix + "/size_XY.pkl")
        # 99 + 380 + 7*5*2 + 2
        print len(train_X[0]), len(train_Y)

    if os.path.exists(util.features_prefix + "/salary_XY.pkl") is False:
        train_Y = list(train["predict_salary"].values)
        label_dict = LabelEncoder().fit(train_Y)
        label_dict_classes = len(label_dict.classes_)
        train_Y = label_dict.transform(train_Y)
        pd.to_pickle([train_X, train_Y], util.features_prefix + "/salary_XY.pkl")
    else:
        [train_X, train_Y] = pd.read_pickle(util.features_prefix + "/salary_XY.pkl")
        99 + 380 + 7*5*2 + 2
        from sklearn.tree import DecisionTreeClassifier

        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(np.array(train_X[:100]), np.array(train_Y[:100]))
        print clf.predict(np.array(train_X[100:200]))
        print train_Y[100:200]
        from sklearn.feature_selection import SelectFromModel

        model = SelectFromModel(clf, prefit=True)
        list_1 = model.get_support()
        for i in range(len(list_1)):
            if list_1[i] == True:
                print i
    print 'pickle end'

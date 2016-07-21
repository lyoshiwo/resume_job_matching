# -*- coding: utf-8 -*-
import util

__author__ = 'lin_eo'
from util import read_json
import datetime
from gensim.models import Word2Vec


def work_time(start_date, end_date):
    start_date = start_date.strip()
    end_date = end_date.strip()
    if u'今' in end_date or 'Present' in end_date or u'其他' in end_date:
        end_date = '2015-05'
    a = str(start_date).split('-')[0]
    b = str(start_date).split('-')[1]
    c = str(end_date).split('-')[0]
    d = str(end_date).split('-')[1]
    d1 = datetime.datetime(int(c), int(d), 1)
    d2 = datetime.datetime(int(a), int(b), 1)
    m = int(round(float((d1 - d2).days) / 30.5, 2) / 3)
    # 工作了多少个季度
    return 'm' + str(m)


def train_w2vector(features, words, sentence_dict_path, word2vec_path):
    file_path_train = util.data_prefix + r'resume_clean.json'
    json_all = read_json(file_path_train)
    sentences = []
    level_zero = [u'size', u'salary']
    level_two = [u'type', u'department', u'industry', u'position_name']
    level_one = [u'major', u'gender']
    dic_all = {}
    count2 = 0
    all = 0
    for i in json_all:
        id_one = i[u'id']
        workExperienceList = i[u'workExperienceList']
        all += 1
        sentence = []
        count = 0
        for w in workExperienceList:
            if w == None:
                continue
            if len(w) > 2:
                count2 += 1
            count += 1
            if count == 1:
                continue
            for s in level_zero:
                try:
                    if w[s] == None:
                        sentence.append(s + u'1')
                    else:
                        sentence.append(s + str(w[s]))
                except Exception, e:
                    sentence.append(s + u'1')

            for t in level_two:
                try:
                    if w[t] == None:
                        sentence.append(u'其他')
                    else:
                        sentence.append(w[t])
                except Exception, e:
                    sentence.append(u'其他')

            try:
                sentence.append(work_time(w['start_date'], w['end_date']))
            except Exception, e:
                sentence.append(u'm4')
        for z in level_one:
            try:
                if i[z] == None:
                    sentence.append(u'其他')
                else:
                    sentence.append(i[z])
            except Exception, e:
                print e.message
        try:
            sentence.append(u'a' + str(i[u'age']))
        except Exception, e:
            sentence.append(u'a24')
        dic_all.setdefault(id_one, sentence)
        sentences.append(sentence)
        # if u'软件工程师' in sentence:
        #     for i in sentence:
        #         print i
        #     exit()

    print len(sentences)
    from util import write_dic
    # 把 id对应的sentences存起来
    write_dic(dic_all, sentence_dict_path)

    # Set values for various parameters
    num_features = features  # Word vector dimensionality
    min_word_count = 1  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = words  # Context window size
    downsampling = 1e-2  # Downsample setting for frequent words

    print "Training Word2Vec model..."
    model = Word2Vec(sentences, workers=num_workers, \
                     size=num_features, min_count=min_word_count, \
                     window=context, sample=downsampling, seed=1, negative=1)

    model.init_sims(replace=True)
    model.save(word2vec_path)
    for w in model.most_similar(u'软件工程师', topn=10):
        print u'软件工程师', w[1], w[0]
    for w in model.most_similar(u'电子商务', topn=10):
        print u'电子商务', w[1], w[0]
    for w in model.most_similar(u'salary3', topn=10):
        print u'salary3', w[1], w[0]


if __name__ == "__main__":
    w2c_f, w2c_w = 30, 14
    # output the id_sentences data to local, it is a dictionary. A sentence means a json resume
    sentence_dict_path = util.txt_prefix + 'id_sentences.pkl'
    word2vec_path = util.txt_prefix + str(w2c_f) + 'features_1minwords_' + str(w2c_w) + 'context.pkl'
    train_w2vector(w2c_f, w2c_w, sentence_dict_path, word2vec_path)

    w2c_f, w2c_w = 10, 14
    # output the id_sentences data to local, it is a dictionary. A sentence means a json resume
    word2vec_path = util.txt_prefix + str(w2c_f) + 'features_1minwords_' + str(w2c_w) + 'context.pkl'
    train_w2vector(w2c_f, w2c_w, sentence_dict_path, word2vec_path)

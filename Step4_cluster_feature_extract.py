# -*- coding: utf-8 -*-
import util

__author__ = 'lin_eo'
from util import read_dict, write_dic
import jieba
import re
import os
from gensim.models import Word2Vec
from gensim.models.ldamodel import LdaModel
from gensim import corpora
from sklearn.cluster import KMeans

if __name__ == "__main__":
    sentence_dict_path = util.txt_prefix + 'id_sentences.pkl'
    if os.path.exists(sentence_dict_path) is False:
        print sentence_dict_path, ' does not exit'
        exit()
    if os.path.exists(util.txt_prefix + 'id_texts.pkl') is False:
        id_sentence = read_dict(sentence_dict_path)
        print len(id_sentence)
        id_text = {}
        for i in id_sentence.keys():
            sentence = id_sentence[i]
            temp = ' '.join(sentence)
            temp = re.sub('-|\\)|\\(|（|/|）', ' ', temp).replace('）', '')
            cut_str = jieba.cut(temp)
            text = " ".join(cut_str)
            text = re.sub(r'\s{2,}', ' ', text)
            id_text.setdefault(i, (text.replace('（', '')).split(' '))
        write_dic(id_text, util.txt_prefix + 'id_texts.pkl')
    id_text = read_dict(util.txt_prefix + 'id_texts.pkl')
    texts = id_text.values()
    features, words = 60, 14
    if os.path.exists(util.txt_prefix + str(features) + 'features_1minwords_' + str(words) + 'context.pkl') is False:
        # Set values for various parameters
        num_features = features  # Word vector dimensionality
        min_word_count = 1  # Minimum word count
        num_workers = 4  # Number of threads to run in parallel
        context = words  # Context window size
        down_sampling = 1e-3  # Down_sample setting for frequent words

        print "Training Word2Vec model..."
        model = Word2Vec(texts, workers=num_workers, \
                         size=num_features, min_count=min_word_count, \
                         window=context, sample=down_sampling, seed=1, negative=1)
        model.init_sims(replace=True)
        word2vec_path = util.txt_prefix + str(features) + 'features_1minwords_' + str(words) + 'context.pkl'
        model.save(word2vec_path)

    topics = 256
    if os.path.exists(util.features_prefix + 'id_lda_' + str(topics) + '.pkl') is False:
        from collections import defaultdict

        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1
        print "frequency finished"
        texts = [[token for token in text if frequency[token] > 3] for text in texts]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        print "text finished"
        lda = LdaModel(corpus, num_topics=topics)
        print "lda finished"


        def turn_list_to_dic(list_temp):
            dic_temp = {}
            for i in list_temp:
                dic_temp.setdefault(i[0], i[1])
            return dic_temp


        dic_write = {}
        count = 0
        for i in range(len(texts)):
            doc_bow = dictionary.doc2bow(texts[i])
            tfidfdict = turn_list_to_dic(lda[doc_bow])
            sorted_tfidf = sorted(tfidfdict.iteritems(), key=lambda d: d[1], reverse=True)
            dic_write.setdefault(id_text.keys()[i], sorted_tfidf[0][0])
            if sorted_tfidf[0][0] > 500:
                print sorted_tfidf[0][0], id_text.values()[i]
        write_dic(dic_write, util.features_prefix + 'id_lda_' + str(topics) + '.pkl')
    topics = 512
    if os.path.exists(util.features_prefix + 'id_lda_' + str(topics) + '.pkl') is False:
        from collections import defaultdict

        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1
        print "frequency finished"
        texts = [[token for token in text if frequency[token] > 3] for text in texts]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        print "text finished"
        lda = LdaModel(corpus, num_topics=topics)
        print "lda finished"


        def turn_list_to_dic(list_temp):
            dic_temp = {}
            for i in list_temp:
                dic_temp.setdefault(i[0], i[1])
            return dic_temp


        dic_write = {}
        count = 0
        for i in range(len(texts)):
            doc_bow = dictionary.doc2bow(texts[i])
            tfidfdict = turn_list_to_dic(lda[doc_bow])
            sorted_tfidf = sorted(tfidfdict.iteritems(), key=lambda d: d[1], reverse=True)
            if sorted_tfidf[0][0] > 500:
                print sorted_tfidf[0][0], id_text.values()[i]
            dic_write.setdefault(id_text.keys()[i], sorted_tfidf[0][0])
        write_dic(dic_write, util.features_prefix + 'id_lda_' + str(topics) + '.pkl')
    if os.path.exists(util.txt_prefix + 'c_v_all.pkl') is False:
        print 'create c_v_all'
        import numpy

        word2vec_path = util.txt_prefix + str(features) + 'features_1minwords_' + str(words) + 'context.pkl'
        model = Word2Vec.load(word2vec_path)
        id_sentence = read_dict(sentence_dict_path)
        sentence = id_sentence.values()
        c_vec = {}
        for s in sentence:
            for section in s:
                used = 0
                temp_vec = numpy.zeros(features)
                if c_vec.has_key(section):
                    continue
                else:
                    try:
                        temp_vec += model[section]
                        c_vec.setdefault(section, temp_vec)
                        print section, temp_vec[0:2]
                    except Exception, e:
                        for ww in jieba.cut(section):
                            try:
                                temp_vec += model[ww]
                                used += 1
                            except Exception, e:
                                continue
                        if used == 0:
                            used = 1
                        c_vec.setdefault(section, temp_vec / used)
                        print section, (temp_vec / used)[0:2]
        write_dic(c_vec, util.txt_prefix + 'c_v_all.pkl')
    k_clusters = 128
    if os.path.exists(util.features_prefix + 'c_k_all_' + str(k_clusters) + '.pkl') is False:
        print 'create c_k_all'
        c_vec = read_dict(util.txt_prefix + 'c_v_all.pkl')
        c_key = c_vec.keys()
        vec_set = c_vec.values()
        KMeans_model = KMeans(n_clusters=k_clusters, n_init=5)
        KMeans_model.fit(vec_set)
        k_labels = KMeans_model.labels_
        dict_temp = {}
        for index in range(len(c_vec)):
            dict_temp.setdefault(c_key[index], k_labels[index])
            if 13 > k_labels[index] > 10:
                print c_key[index], k_labels[index]
        print len(dict_temp)
        write_dic(dict_temp, util.features_prefix + 'c_k_all_' + str(k_clusters) + '.pkl')
    k_clusters = 64
    if os.path.exists(util.features_prefix + 'c_k_all_' + str(k_clusters) + '.pkl') is False:
        print 'create c_k_all'
        c_vec = read_dict(util.txt_prefix + 'c_v_all.pkl')
        c_key = c_vec.keys()
        vec_set = c_vec.values()
        KMeans_model = KMeans(n_clusters=k_clusters, n_init=5)
        KMeans_model.fit(vec_set)
        k_labels = KMeans_model.labels_
        dict_temp = {}
        for index in range(len(c_vec)):
            dict_temp.setdefault(c_key[index], k_labels[index])
            if 13 > k_labels[index] > 10:
                print c_key[index], k_labels[index]
        print len(dict_temp)
        write_dic(dict_temp, util.features_prefix + 'c_k_all_' + str(k_clusters) + '.pkl')
c_k_all = read_dict(util.features_prefix + 'c_k_all_' + str(k_clusters) + '.pkl')
set_cluster = set(c_k_all.values())
set_cluster = list(set_cluster)
flg = range(10)
for i in flg:
    for k in c_k_all.keys():
        if set_cluster[i] == c_k_all[k]:
            print set_cluster[i], k

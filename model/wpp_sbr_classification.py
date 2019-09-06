from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models.word2vec import Word2Vec
from data_process import data_filter
import numpy as np
import pandas as pd
from pandas import DataFrame
import config
from data_process import preprocess_br
from sklearn_evaluation import plot, table
from SecurityBRClassifier import SecurityClassifier
from SecurityBRClassifierM import SecurityClassifierM
import os,re
import pickle,json
from collections import defaultdict

# 计算句子的向量
def to_review_vector(words, model):
    array = np.array([model[w] for w in words if w in model])
    return pd.Series(array.mean(axis=0))

# sigle model predict
word2vec_model = Word2Vec.load(config.EMBEDDING_PATH)
def wpp_sbr_predict(df, df_train, train_info):

    X_train = df_train.apply(lambda x: to_review_vector(x.summary + x.description, word2vec_model),
                             axis=1)
    y_train = df_train.Security

    df_test = DataFrame(df, index=range(int(len(df) / 2), len(df)))
    print(len(df_test))
    X_test = df_test.apply(lambda x: to_review_vector(x.summary + x.description, word2vec_model),
                           axis=1)
    y_test = df_test.Security

    # X_train, X_test, y_train, y_test = train_test_split(content, label, test_size=0.4, random_state=42)
    sc = SecurityClassifier(train_info[0])
    sc.train(X_train, y_train)

    y_pred = sc.predict_b(X_test)
    if train_info[0] == 'Blend':
        y_pred = list(np.argmax(y_pred, axis=1))

    result = sc.evaluate_b(y_test, y_pred)
    print(result)
    y_score = sc.predict_p(X_test)
    # sc.evaluate_p(y_test,y_score)
    data = {'result': result, 'y_pred': y_pred, 'y_score': y_score, 'y': y_test}

    # save_result_rsms(data, train_info)
    # save_result(data, train_info)

def save_result(data, train_info):
    result_path = os.path.join('..', 'resources', 'g_measure')
    file_list = os.listdir(result_path)
    count = 0
    for file_name in file_list:
        fn_split = file_name.split('_')
        # [classifier,project,filter_way,'word2vec','rank_way']
        if fn_split[0] == 'wpp' and fn_split[1] == train_info[0] and fn_split[2] == train_info[1] and fn_split[3] == \
                train_info[2] and fn_split[4] == train_info[3]:
            if float(fn_split[-1]) < data['result'][-1]:
                os.remove(os.path.join(result_path, file_name))
                # store
                fn_split[-1] = str(data['result'][-1])
                new_file_name = '_'.join(fn_split)
                f = open(os.path.join(result_path, new_file_name), 'wb')
                pickle.dump(data, f)
            count = count + 1
    if count == 0:
        file_name = 'wpp' + '_' + train_info[0] + '_' + train_info[1] + '_' + train_info[2] + '_' + train_info[
            3] + '_' + str(data['result'][-1])
        f = open(os.path.join(result_path, file_name), 'wb')
        pickle.dump(data, f)

# store the comparative result of rs-filter and ms-filter
def save_result_rsms(data,train_info):
    train_way = re.search(r'(.+)s', train_info[2]).group(0)
    result_path = os.path.join('..', 'resources', 'g_measure_wpp_rs',train_info[0] + '_' + train_info[1]+'_10times_'+train_way)
    if not os.path.exists(result_path):
        exist_data=defaultdict(list)
    else:
        f = open(result_path, 'r')
        exist_data = json.load(f)
        f.close()
    key=train_info[2]
    if key in exist_data:
        exist_data[key].append(data['result'][-1])
    else:
        exist_data[key]=[data['result'][-1]]

    f = open(result_path, 'w')
    json.dump(exist_data,f)
    f.close()


# multiple model predicts
def wpp_sbr_predicts(df, df_train):
    sc = Word2Vec.load(config.EMBEDDING_PATH)
    X_train = df_train.apply(lambda x: to_review_vector(x.summary + x.description, sc),
                             axis=1)
    y_train = df_train.Security

    df_test = DataFrame(df, index=range(int(len(df) / 2), len(df)))
    print(len(df_test))
    X_test = df_test.apply(lambda x: to_review_vector(x.summary + x.description, sc),
                           axis=1)
    y_test = df_test.Security

    path = os.path.join('..', 'resources', 'result', project)
    # y_score = sc.predict_p(X_test)
    # sc.evaluate_p(y_test,y_score)
    sc = SecurityClassifierM(['MLP', 'NB', 'SVM', 'LR', 'KNN'])
    sc.train(X_train, y_train)

    y_pred = sc.predict_b(X_test)
    result = sc.evaluate_b(y_test, y_pred)
    print(result)
    y_score = sc.predict_p(X_test)
    sc.evaluate_p(y_test, y_score, path)


# single model predict
def wpp_experient_result():
    train_way = 'wpp'
    # projects = ['chromium', 'wicket', 'ambari','camel', 'derby'],['chromium_large','mozilla']
    projects = ['chromium', 'wicket', 'ambari','camel', 'derby']
    # ['ms', 'rs']
    filter_ways = ['ms']
    rw='BM25F'
    # times_list = list(range(1, 10))
    times_list = [9]
    # 'RFCV', 'LRCV', 'KNNCV','LR', 'MLP', 'SVM', 'NB', 'RF','KNN' ,'NBCV','LRCV', 'SVMCV','MLPCV','RFCV', 'KNNCV'
    classifiers = ['SVM']
    for i in range(1):
        for project in projects:
            for classifier in classifiers:
                for filter_way in filter_ways:
                    if filter_way=='ns':
                        times=None
                        print([project, classifier, filter_way, times])
                        df, df_train = data_filter(project, times, filter_way, train_way,rank_way=rw)
                        wpp_sbr_predict(df, df_train, [classifier, project, filter_way + str(times), 'word2vec',rw])
                        continue
                    for times in times_list:
                        if times==10 and project=='derby':
                            continue
                        print([project, classifier, filter_way, times])
                        df, df_train = data_filter(project, times, filter_way, train_way,rank_way=rw)
                        wpp_sbr_predict(df, df_train, [classifier, project, filter_way + str(times), 'word2vec',rw])


if __name__ == '__main__':
    print('wpp security bug report classification training...')
    # single project test
    # project: camel ambari wicket derby chromium
    project = 'ambari'
    filter_way = 'rs'
    train_way = 'wpp'
    times = 4
    classifier = 'SVM'
    # df, df_train = data_filter(project, times, filter_way, train_way)
    # wpp_sbr_predict(df, df_train, [classifier, project, filter_way + str(times), 'word2vec'])

    # form: ms(Multiple selection), rs='roulettewheel selection; ns='no filter'(all data),num is multiple
    # df, df_train = data_filter(project, times, filter_way,train_way)
    # LR MLP SVM NB RF KNN SVMCV MLPCV RFCV
    # classifier='SVMCV'
    # wpp_sbr_predict(df, df_train, [classifier, project, filter_way + str(times), 'word2vec'])
    # wpp_sbr_predicts(df, df_train)
    for i in range(1):
        wpp_experient_result()





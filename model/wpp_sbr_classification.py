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
    X_test = df_test.apply(lambda x: to_review_vector(x.summary + x.description, word2vec_model),
                           axis=1)
    y_test = df_test.Security

    sc = SecurityClassifier(train_info[0])
    sc.train(X_train, y_train)

    y_pred = sc.predict_b(X_test)

    result = sc.evaluate_b(y_test, y_pred)
    print(result)
    y_score = sc.predict_p(X_test)
    data = {'result': result, 'y_pred': y_pred, 'y_score': y_score, 'y': y_test}


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



# single model predict
def wpp_experient_result():
    train_way = 'wpp'
    # ['chromium_large','mozilla']
    projects = ['chromium', 'wicket', 'ambari','camel', 'derby']
    filter_ways = ['ms', 'rs']
    rw='BM25F'
    times_list = list(range(1, 10))
    classifiers = ['RFCV', 'LRCV', 'KNNCV','LR', 'MLP', 'SVM', 'NB', 'RF','KNN' ,'NBCV','LRCV', 'SVMCV','MLPCV','RFCV', 'KNNCV']
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
    wpp_experient_result()





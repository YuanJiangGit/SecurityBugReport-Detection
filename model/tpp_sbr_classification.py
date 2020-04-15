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
from wpp_sbr_classification import to_review_vector
from data_process import clean_pandas
import os
import pickle


# sigle model predict
def tpp_sbr_predict(train_info):
    # [classifier, project, filter_way, times]
    modle_name = train_info[0]
    project = train_info[1]
    filter_way = train_info[2]
    train_way = 'tpp'
    times = train_info[3]

    word2vec_model = Word2Vec.load(config.EMBEDDING_PATH)
    datasets = {
        'ambari': 'Ambari.csv',
        'camel': 'Camel.csv',
        'chromium': 'Chromium.csv',
        'derby': 'Derby.csv',
        'wicket': 'Wicket.csv'
    }
    if project not in datasets:
        raise ValueError

    source_model = {}
    for key, value in datasets.items():
        if key == project:
            data_file = os.path.join('..', 'resources', datasets[key])
            df = pd.read_csv(data_file, sep=',', encoding='ISO-8859-1')
            df = clean_pandas(df)
            df_test = DataFrame(df, index=range(int(len(df) / 2), len(df)))
            print(len(df_test))
            X_test = df_test.apply(lambda x: to_review_vector(x.summary + x.description, word2vec_model), axis=1)
            y_test = df_test.Security
        else:
            _, df_train = data_filter(key, times, filter_way, train_way)
            X_train = df_train.apply(
                lambda x: to_review_vector(x.summary + x.description, word2vec_model),
                axis=1)
            y_train = df_train.Security
            sc = SecurityClassifier(modle_name)
            sc.train(X_train, y_train)
            source_model[key] = sc

    best_result = 0.0
    for source_project, sc in source_model.items():
        y_pred = sc.predict_b(X_test)
        print('The source file：%s' % source_project)
        result = sc.evaluate_b(y_test, y_pred)
        print(result)
        y_score = sc.predict_p(X_test)
        if best_result < result[-1]:
            # sc.evaluate_p(y_test,y_score)
            data = {'result': result, 'y_pred': y_pred, 'y_score': y_score, 'y': y_test, 'project': source_project}
            best_result = result[-1]


def tpp_experient_result():
    # projects = ['camel', 'ambari', 'wicket', 'derby', 'chromium']
    projects = ['chromium']
    filter_ways = ['ns']
    times_list = list(range(1, 11))
    classifiers = ['LR', 'MLP', 'SVM', 'NB', 'RF', 'KNN', 'SVMCV', 'MLPCV', 'RFCV', 'LRCV', 'KNNCV']
    for project in projects:
        for classifier in classifiers:
            for filter_way in filter_ways:
                if filter_way == 'ns':
                    times = None
                    print([project, classifier, filter_way, times])
                    tpp_sbr_predict([classifier, project, filter_way, times])
                    continue
                for times in times_list:
                    print([project, classifier, filter_way, times])
                    tpp_sbr_predict([classifier, project, filter_way, times])


if __name__ == '__main__':
    print('tpp security bug report classification training...')
    tpp_experient_result()
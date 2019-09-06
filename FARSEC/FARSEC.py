import pre
import os
from pandas import DataFrame, Series
from gensim.sklearn_api import TfIdfTransformer
import numpy as np
from gensim import corpora, models
from SecurityBRClassifier import SecurityClassifier
import pickle
import gensim
from SecurityBRClassifierM import SecurityClassifierM


# how resistant is FARSEC to mislabelled bug reports
def resistant_cap():
    # projects = ['chromium','wicket','ambari','camel','derby']
    projects = ['chromium', 'ambari', 'camel', 'derby']
    columns = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    df_noise = DataFrame(columns=columns, index=projects)
    df_sbr = DataFrame(columns=columns, index=projects)

    for project in projects:
        filter = pre.Filtering()
        # read data
        filter.read_project(project)

        dict_noise = {}
        dict_sbr = {}
        for percent in columns:
            times = 10
            # id list of noise issue bug report
            noise_issue_id = filter.make_noise(percent)
            # 找到安全相关关键词
            filter.findSRW()
            print("find security related keywords succeed")
            filter_issue_id = filter.farsec(support='farsecsq')
            train_noise_issue = set(noise_issue_id).intersection(set(filter_issue_id))
            num_noise_issue, len_sbr = len(train_noise_issue), len(filter.SBR)
            print('the number of noise issue report is %s' % num_noise_issue)
            dict_noise[percent] = num_noise_issue
            dict_sbr[percent] = len_sbr
        df_noise.loc[project] = Series(dict_noise)
        df_sbr.loc[project] = Series(dict_sbr)
    result_dir = '../resources/result/resistant2FN'
    df_noise.to_csv(os.path.join(result_dir, 'FARSEC_noise.csv'))
    df_sbr.to_csv(os.path.join(result_dir, 'FARSEC_sbr.csv'))



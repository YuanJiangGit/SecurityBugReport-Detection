
import os
import pandas as pd
import math
from pandas import DataFrame,Series
import BugReport
from data_process import clean_pandas,split_report,BM25F_filter,VSM_filter
# filter NSBR
def data_filter(project, times, form, train_way,percent,rank_way='BM25F'):
    datasets = {
        'ambari': 'Ambari2.csv',
        'camel': 'Camel2.csv',
        'chromium': 'Chromium.csv',
        'derby': 'Derby2.csv',
        'wicket': 'Wicket2.csv'
    }
    if project not in datasets:
        raise ValueError

    data_file = os.path.join('..', 'resources', datasets[project])
    pandas_data_file=os.path.join('..', 'resources','pandas_data',project)

    df_all = pd.read_csv(data_file, sep=',', encoding='ISO-8859-1')
    # specialized processing with chromium
    path = os.path.join('..', 'resources', 'Chromium2.csv')
    if project == 'chromium':
        if not os.path.exists(path):
            df_all['summary'] = df_all.apply(lambda x: split_report(x.report, 'summary'), axis=1)
            df_all['description'] = df_all.apply(lambda x: split_report(x.report, 'description'), axis=1)
            df_all.to_csv(path, encoding='utf-8')
        else:
            df_all = pd.read_csv(path)
    # clean the textual fileds
    df_all = clean_pandas(df_all)
    df_all.to_pickle(pandas_data_file)

    if train_way=='wpp':
        df = DataFrame(df_all, index=range(int(len(df_all) / 2)))
    elif train_way=='tpp':
        df=DataFrame(df_all)
    else:
        return None
    print(len(df))

    # randomly selected n% SBRS and artificially change their labels from security to non-security
    df_sbr = df[df.Security == 1]
    # df.Security.apply(lambda x: 0 if x.security==1 else 0,axis=1)
    print('the number of sbr before the change is %s'%len(df_sbr))
    noise_issue_id=[]
    # the number of noise data
    noise_num=math.ceil(len(df_sbr)*percent)

    for i in range(noise_num):
        issue_id=df_sbr.iloc[i].get('issue_id')
        noise_issue_id.append(issue_id)
    # update the value of label(change SBR to NSBR)
    df.Security=df.apply(lambda x:0 if x.issue_id in noise_issue_id else x.Security, axis = 1)

    df_sbr = df[df.Security == 1]
    # add FNs to the golden set
    print('the length of SBR after the change is %s' % len(df_sbr))

    path = os.path.join('..', 'resources', rank_way, train_way+'_'+project)
    if rank_way=='BM25F':
        filter_issue_id=BM25F_filter(df,path,times,form,len(df_sbr))
    elif rank_way=='VSM':
        # vsm filter way
        filter_issue_id=VSM_filter(df,path,times,form,len(df_sbr))
    else:
        return None
    # the intersection between filter_issue_id and noise_issue_id
    train_noise_issue=set(noise_issue_id).intersection(set(filter_issue_id))
    return len(train_noise_issue),len(df_sbr)


def validate_filter_capab():
    train_way = 'wpp'
    # projects = ['chromium','wicket','ambari','camel','derby']
    projects=['chromium','ambari','camel','derby']
    columns=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    df_noise = DataFrame(columns=columns, index=projects)
    df_sbr= DataFrame(columns=columns, index=projects)

    for project in projects:

        dict_noise={}
        dict_sbr={}
        for percent in columns:
            times = 10
            num_noise_issue,len_sbr= data_filter(project, times, 'ms', train_way,percent)
            print('the number of noise issue report is %s'%num_noise_issue)
            dict_noise[percent]=num_noise_issue
            dict_sbr[percent]=len_sbr
        df_noise.loc[project]=Series(dict_noise)
        df_sbr.loc[project]=Series(dict_sbr)
    result_dir = '../resources/result/resistant2FN'
    df_noise.to_csv(os.path.join(result_dir, 'LTRWES_noise.csv'))
    df_sbr.to_csv(os.path.join(result_dir, 'LTRWES_sbr.csv'))

if __name__ == '__main__':
    validate_filter_capab()
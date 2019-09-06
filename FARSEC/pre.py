
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer 
from gensim import corpora 
import csv
import string
import numpy as np
import math
import os
import pandas as pd
from pandas import DataFrame
import operator
from data_process import split_report,clean_pandas


class Filtering:

    def __init__(self):
        self.cross_word = None
        self.SBRmat = None
        self.NSBRmat = None
        self.SBR = None
        self.NSBR = None
        self.dictionary =None
        self.BR = None
        self.BR_all=None

    def read_project(self,project):
        datasets = {
            'ambari': 'Ambari2.csv',
            'camel': 'Camel2.csv',
            'chromium': 'Chromium.csv',
            'derby': 'Derby2.csv',
            'wicket': 'Wicket2.csv',
            'mozilla':'mozilla_large.csv',
            'chromium_large':'chromium_large.csv'
        }
        if project not in datasets:
            raise ValueError

        data_file = os.path.join('..', 'resources', datasets[project])
        pandas_data_file = os.path.join('..', 'resources', 'pandas_data', project)
        if os.path.exists(pandas_data_file):
            df_all = pd.read_pickle(pandas_data_file)
        else:
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
        # the form 1/2 as training data
        df_train = DataFrame(df_all, index=range(int(len(df_all) / 4)))
        self.SBR = df_train[df_train['Security'] == 1]
        self.NSBR = df_train[df_train['Security'] == 0]
        self.BR_all=df_all
        self.BR=df_train

    '''
    make noise by change the security label to non-security label in accordance with a certain proportion
    '''
    def make_noise(self,percent):
        df=self.BR.copy(deep=True)
        # randomly selected n% SBRS and artificially change their labels from security to non-security
        df_sbr = df[df.Security == 1]
        # df.Security.apply(lambda x: 0 if x.security==1 else 0,axis=1)
        print('the number of sbr before the change is %s' % len(df_sbr))
        noise_issue_id = []
        # the number of noise data
        noise_num = math.ceil(len(df_sbr) * percent)
        # 向上取整
        for i in range(noise_num):
            issue_id = df_sbr.iloc[i].get('issue_id')
            noise_issue_id.append(issue_id)
        # update the value of label(change SBR to NSBR)
        df.Security = df.apply(lambda x: 0 if x.issue_id in noise_issue_id else x.Security, axis=1)

        df_sbr = df[df.Security == 1]
        # add FNs to the golden set
        print('the length of SBR after the change is %s' % len(df_sbr))
        self.SBR = df[df['Security'] == 1]
        self.NSBR = df[df['Security'] == 0]
        return noise_issue_id


    def findSRW(self):
        # 对应论文3.1 Identifying Security Related Keywords

        SBR = self.SBR['summary']+self.SBR['description']
        # 提取SBR所有的术语,用词袋模型转化为向量
        self.dictionary = corpora.Dictionary(SBR)
        # 论文中提到去掉一些低频无用词语，并给了关于这些词语的网址，此处应补充

        # 词袋将文本转化为向量,第一步产生的是一个向量队列,(1,2)表示词语1出现2次,一句文本又一组词频表示,需要转化为矩阵
        SBR = [self.dictionary.doc2bow(doc) for doc in SBR]
        SBRmat = self.makematrix(SBR, len(self.dictionary))

        tf_idf = self.tf_idf(SBRmat)

        '''tf_idf选出前Top100,
        此处有个疑问:
        举个例子, 一个词语在多个文档中出现,和一个词语在较少文本中出现,tf-idf矩阵为:
                t1      t2
        br1     0.2     0.8
        br2     0.2     0.05
        ...     ...     ...
        brn     0.2     0.1
        t1在大多数文本中出现, tf-idf分数低, 但是在各个文本都有分布, t2在很少文本中出现, 但能出现比较大的数字
        最终偏向哪一类型的词语值得考究

        程序运行时发现, 论文中筛选出来的SRW在这个程序中tf-idf反而分数低,可能跟此处的求和有关
        '''
        terms = np.sum(tf_idf, axis=0)
        terms = np.argsort(terms)[-300:]

        # 字典保留前100个词语
        self.dictionary.filter_tokens(good_ids=terms.tolist())
        self.cross_word = self.dictionary.token2id
        # print("there are the security related word", self.cross_word)

        return self.dictionary

    def farsec(self,times=10, support='farsecsq'):

        M = self.ScoreKeywords(support=support)
        BRscore = self.ScoreBugReports(M)
        # sort BR according by BRscore in descending order
        br_score_sorted,issue_id_sorted=zip(*sorted(zip(BRscore,self.NSBR.issue_id),key=operator.itemgetter(0),reverse=True))

        # selected the top-k NSBR as noise set
        # 用0.75划分，高于0.75的为噪音NSBR
        # BRscore[BRscore < 0.75] = 0
        # BRscore[BRscore > 0] = 1
        # print("remain : ", BRscore.shape[0] - BRscore.sum())

        to_train_in_this = 1
        if type(times)==int:
            return issue_id_sorted[:times*len(self.SBR)]
        else:
            BRscore_array=np.array(BRscore)
            bug_index=np.where(BRscore_array<=0.75)
            return bug_index


    def tf(self, D):
        # 每行的最大值,结果为n行一列矩阵
        max_w = np.array([D.max(axis=1)]).T

        tf = 0.5 + (0.5 * D) / max_w

        ''' tf 格式:

            t1 ... tn
        br1 
        ... 
        brn 

        '''
        return tf

    def idf(self, D):
        # 文件总数N,
        N = D.shape[0]
        D[D > 0] = 1

        D = D.sum(axis=0)
        idf = np.log(N / D)

        ''' idf 格式:

        t1   ... tn
        idf1 ... idfn

        '''
        return idf

    def tf_idf(self, D):

        tf = self.tf(D)
        idf = self.idf(D)

        tf_idf = tf * idf
        ''' tf-idf 格式:

             t1 ... tn
        br1 
        ...   tf-idf
        brn 

        '''
        return tf_idf

    def ScoreKeywords(self, support):
        dictionary = self.dictionary

        # 对应论文W，安全相关词的个数
        W = len(self.dictionary)
        SBR = self.SBR['summary']+self.SBR['description']
        NSBR = self.NSBR['summary']+self.NSBR['description']
        dictionary.add_documents(SBR)
        dictionary.add_documents(NSBR)
        # print('all terms : the security related keywords', len(self.dictionary), ':', W)
        S = [dictionary.doc2bow(doc) for doc in SBR]
        S = self.makematrix(S, len(dictionary))
        SBR = S[:, :W]
        self.SBRmat = SBR

        NS = [dictionary.doc2bow(doc) for doc in NSBR]
        NS = self.makematrix(NS, len(dictionary))
        NSBR = NS[:, :W]
        self.NSBRmat = NSBR

        S = S.sum()
        NS = NS.sum()

        SBR = SBR.sum(axis=0)
        if 'clni' in support:
            NSBR = self.CLNI()
        if 'sq' in support:
            SBR *= SBR
        elif 'two' in support:
            SBR *= 2
        SBR /= S
        SBR[SBR > 1] = 1  # 不能大于１

        NSBR = NSBR.sum(axis=0)
        NSBR /= NS
        NSBR[NSBR > 1] = 1

        M = SBR / (SBR + NSBR)
        M[M > 0.99] = 0.99
        M[M < 0.01] = 0.01
        # print("key words score : ")
        # print(M)
        return M

    def ScoreBugReports(self, M):
        NSBR = self.NSBRmat
        NSBR[NSBR > 1] = 1

        Mstar = NSBR * M

        Mstar[Mstar == 0] = 1
        Mstar = Mstar.prod(1)

        Mstar[Mstar == 1] = 0

        Mquote = NSBR * (1 - M)
        Mquote[Mquote == 0] = 1
        Mquote = Mquote.prod(1)

        return Mstar / (Mstar + Mquote)

    def CLNI(self):
        NSBR = self.NSBRmat
        SBR = self.SBRmat

        del_count = 0
        e = 0
        while e < 0.99:
            disNN = self.EuclideanDistances(NSBR, NSBR)
            disNN.sort()
            disNS = self.EuclideanDistances(NSBR, SBR)
            disNS.sort()
            Noise = disNN[:, 4] - disNS[:, 0]
            index = []
            for i in range(Noise.shape[0]):
                if Noise[i] > 0:
                    index.append(i)
            delect = len(index)
            if del_count + delect == 0:
                e = 1
            else:
                e = del_count / (del_count + delect)
            del_count = del_count + delect
            NSBR = np.delete(NSBR, index, axis=0)
        self.NSBRmat = NSBR
        print('CLNI delect : ', del_count)
        return NSBR

    def EuclideanDistances(self, A, B):
        AB = A.dot(B.T)
        Asq = np.array([(A ** 2).sum(axis=1)]).T  # A行1列
        Bsq = (B ** 2).sum(axis=1)  # 1行B列
        distance = -2 * AB + Asq + Bsq
        return distance

    def makematrix(self, data, lenth):

        matrix = np.zeros((len(data), lenth))

        for row in range(len(data)):
            for col in data[row]:
                matrix[row, col[0]] = col[1]

        return matrix
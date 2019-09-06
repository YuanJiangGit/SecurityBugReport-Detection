import os
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import re
from BugReport import BugReport
import itertools
import nltk
import string, random
import pickle
from REP import REP
from functools import cmp_to_key
from collections import defaultdict
from pandas import DataFrame
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
import operator
# Read key words of java program language
fp = open(r'../resources/keyword.txt')
keyword = []
for line in fp.readlines():
    keyword.append(line.replace("\n", ""))
fp.close()
remove_punctuation_map = dict((ord(char), ' ' + char + ' ') for char in string.punctuation)
remove_number_map = dict((ord(char), " ") for char in string.digits)


# Remove code snippet
def remove_code(text):
    temp = text.translate(remove_punctuation_map)
    temp = temp.translate(remove_number_map)
    list_words = word_tokenize(temp)

    all = len(list_words)
    if (all == 0):
        return ''
    keywords = len([w for w in list_words if w in keyword])
    if (1.0 * keywords / all > 0.3):
        return ''
    else:
        return text


# Processing the text of BugReport
def preprocess_br(raw_description):
    # 1. Remove \r
    current_desc = raw_description.replace('\r', ' ')
    # 2. Remove URLs
    current_desc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                          current_desc)
    # 3. Remove Stack Trace
    start_loc = current_desc.find("Stack trace:")
    current_desc = current_desc[:start_loc]
    # 3.5 Remove Issue id : (only exists in chromium, others project can commented out)
    current_desc = re.sub(r'Issue (.*) : ', '', current_desc)
    # 4. Remove hex code
    current_desc = re.sub(r'(\w*)0x\w+', '', current_desc)
    # 5. Remove code snippet
    # current_desc=remove_code(current_desc)
    # 6. Change to lower case
    current_desc = current_desc.lower()
    # 7. only letters
    letters_only = re.sub("[^a-zA-Z\.]", " ", current_desc)
    current_desc = re.sub("\.(?!((c|h|cpp|py)\s+$))", " ", letters_only)
    # 8. Tokenize
    current_desc_tokens = nltk.word_tokenize(current_desc)
    # 9. Remove stop words
    stopwords = set(nltk.corpus.stopwords.words('english'))
    meaningful_words = [w for w in current_desc_tokens if not w in stopwords]
    # 10. Stemming
    snowball = SnowballStemmer("english")
    stems = [snowball.stem(w) for w in meaningful_words]
    return stems


def getBR(bug_series, type=1):
    # 实例化类对象
    br = BugReport()
    # 处理一行最末尾的‘\n’
    # bug_series = bug_series.replace('\n', ',').split(',')

    br.bugID = bug_series.get('issue_id')
    # summ = bug_series.get('summary')
    # br.summ = summ.split(' ')
    br.summ = bug_series.get('summary')

    pre_term = ' '
    for index, term in enumerate(br.summ):
        if term == '': continue
        br.summ_unigram.append(term)
        br.summ_times_unigram[term] = br.summ_times_unigram[term] + 1
        br.totalWords_unigram.append(term)
        if type == 1:
            BugReport.word_inBR_unigram[term].add(br.bugID)
        if index > 0:
            temp_bigram = pre_term + ' ' + term
            br.summ_bigram.append(temp_bigram)
            br.summ_times_bigram[temp_bigram] = br.summ_times_bigram[temp_bigram] + 1
            if type == 1:
                BugReport.word_inBR_bigram[temp_bigram].add(br.bugID)
            br.totalWords_bigram.append(temp_bigram)
        pre_term = term
    br.desc = bug_series.get('description')
    for index, term in enumerate(br.desc):
        if term == '': continue
        br.desc_unigram.append(term)
        br.desc_times_unigram[term] = br.desc_times_unigram[term] + 1
        br.totalWords_unigram.append(term)
        if type == 1:
            BugReport.word_inBR_unigram[term].add(br.bugID)
        if index > 0:
            temp_bigram = pre_term + ' ' + term
            br.desc_bigram.append(temp_bigram)
            br.desc_times_bigram[temp_bigram] = br.desc_times_bigram[temp_bigram] + 1
            if type == 1:
                BugReport.word_inBR_bigram[temp_bigram].add(br.bugID)
            br.totalWords_bigram.append(temp_bigram)
        pre_term = term

    BugReport.lenSum_unigram += len(br.summ_unigram)
    BugReport.lenSum_bigram += len(br.summ_bigram)
    BugReport.lenDesc_unigram += len(br.desc_unigram)
    BugReport.lenDesc_bigram += len(br.desc_bigram)

    return br


def dealNan(x):
    if type(x) == float or type(x) == list:
        x = ' '
    return x


def clean_pandas(data):
    data['summary'] = data.summary.apply(dealNan)
    data['description'] = data.description.apply(dealNan)
    data['summary'] = data['summary'].map(lambda x: preprocess_br(x))
    data['description'] = data['description'].map(lambda x: preprocess_br(x))

    return data


def preprocess_strict(raw_description):
    description = rm_cite(raw_description)
    # 2. Remove urls
    text = re.sub("((mailto\:|(news|(ht|f)tp(s?))\://){1}\S+)", " ", description)
    # 3. Remove Stack Trace
    start_loc = text.find("Stack trace:")
    text = text[:start_loc]
    # 4. Remove hex code
    text = re.sub(r'(\w+)0x\w+', '', text)
    # 5. Remove non-letters
    # letters_only = re.sub("[^a-zA-Z_/\-\.]", " ", description_text)
    letters_only = re.sub("[^a-zA-Z\.]", " ", text)
    letters_only = re.sub("\.(?!((c|h|cpp|py)\s+$))", " ", letters_only)
    # 6. Convert to lower case, tokenize
    words = [word for sent in nltk.sent_tokenize(letters_only.lower()) for word in nltk.word_tokenize(sent)]
    # 7. Remove stop words
    stopwords = set(nltk.corpus.stopwords.words('english'))
    meaningful_words = [w for w in words if not w in stopwords]
    # 8. Stemming
    snowball = SnowballStemmer("english")
    stems = [snowball.stem(w) for w in meaningful_words]
    return " ".join(stems)

def rm_cite(raw_text):
    lst = []
    iscite = False
    lines = raw_text.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("(In reply to comment"):
            iscite = True
            lst.append(idx)
        elif line.startswith(">"):
            if iscite == True:
                lst.append(idx)
        else:
            iscite = False
    return '\n'.join([item[1] for item in filter(lambda x: x[0] not in lst, enumerate(lines))])

def resCmp(a, b):
    return a.REP - b.REP

# specialized processing with chromium
def split_report(raw_text, tag):
    text_list = re.split(r'[;?\n]\s*', raw_text)
    summary = text_list.pop(0)
    while (len(summary) < 35 and len(text_list)>0):
        summary = summary + text_list.pop(0)
    if tag == 'summary':
        return summary
    else:
        return ' '.join(text_list)

# compute the similarity between NSBRs and SBRs
def bm25F_sort(df):
    # stores instances as BugReport object
    for row_num in range(df.iloc[:, 0].size):
        if df.iloc[row_num].get('Security') == 0:
            BugReport.trainBR.append(getBR(df.iloc[row_num]))
        if df.iloc[row_num].get('Security') == 1:
            BugReport.testBR.append(getBR(df.iloc[row_num]))
    print(len(BugReport.trainBR))
    print(len(BugReport.testBR))
    rep = REP()

    relevance_list=[]
    for index, testBr in enumerate(BugReport.testBR):
        # print(index)
        bmRes = rep.getBM25F_text(BugReport.testBR[index], index)
        relevance_list.append([x.REP for x in bmRes])
    relevance_matric=np.array(relevance_list)
    # compute mean value of each row (axis=1)
    mean_array=np.mean(relevance_matric, axis=0)
    # compute std value of each row (axis=1)
    std_array=np.std(relevance_matric,axis=0)

    for i, rtemp in enumerate(bmRes):
        # bmRes[i].REP = mean_array[i] + std_array[i]
        bmRes[i].REP = mean_array[i]
    # sorted in ascending
    bmRes = sorted(bmRes, key=cmp_to_key(resCmp))

    # global variable, it needs to be emptied
    init_globa_variable()
    return bmRes

def init_globa_variable():
    BugReport.word_inBR_unigram=defaultdict(set)
    BugReport.word_inBR_bigram=defaultdict(set)
    BugReport.lenSum_unigram = float(0)
    BugReport.lenDesc_unigram = float(0)
    BugReport.lenSum_bigram = float(0)
    BugReport.lenDesc_bigram =float(0)
    BugReport.trainBR=[]
    BugReport.testBR=[]

# filter NSBR
def data_filter(project, times, form, train_way,rank_way='BM25F'):
    datasets = {
        'ambari': 'Ambari2.csv',
        'camel': 'Camel2.csv',
        'chromium': 'Chromium.csv',
        'derby': 'Derby2.csv',
        'wicket': 'Wicket2.csv',
        'chromium_large': 'chromium_large.csv',
        'mozilla': 'mozilla_merge_update_process2.csv'
    }
    if project not in datasets:
        raise ValueError

    data_file = os.path.join('..', 'resources', datasets[project])
    pandas_data_file=os.path.join('..', 'resources','pandas_data',project)
    if os.path.exists(pandas_data_file):
        df_all=pd.read_pickle(pandas_data_file)
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
        if project == 'chromium_large' or project == 'mozilla':
            df_all['summary'] = df_all['bug_title_pro'].apply(lambda x: x.split(' '))
            df_all['description'] = df_all['bug_description_pro'].apply(lambda x: x.strip().split(' '))
        else:
            # clean the textual fileds
            df_all = clean_pandas(df_all)
        df_all.to_pickle(pandas_data_file)
    # take the first half as candidate filter data
    # df=df_all.loc[0:int(len(df_all)/2)]
    if train_way=='wpp':
        df = DataFrame(df_all, index=range(int(len(df_all) / 2)))
    elif train_way=='tpp':
        df=DataFrame(df_all)
    else:
        return None
    print(len(df))
    # print the information of columns
    # print(df.columns)

    df_sbr = df[df.Security == 1]
    path = os.path.join('..', 'resources', rank_way, train_way+'_'+project)
    if rank_way=='BM25F':
        issue_id_list=BM25F_filter(df,path,times,form,len(df_sbr))

    # issue_id_list.extend(temp.bugID for temp in BugReport.testBR)
    # the result set(pandas dataframe) of Training data
    df_filter = df[df.issue_id.map(lambda x: x in issue_id_list)]
    df_train = pd.concat([df_filter, df_sbr], ignore_index=True)
    return df_all, df_train



'''
rank model is BM25F
input: dataset(pandas form), intermediate data path, ratio of NSBRs to SBRs, fiter way(ms or ns or none), the number of SBRs
return: the id of bug reports
'''
def BM25F_filter(df, bmres_path,times,form,len_sbr):
    bmRes = bm25F_sort(df)
    # take the top-k elements(the size of k is twice that of
    # bmRes=bmRes[:len(BugReport.testBR)*5]
    bmRes = bm25F_filter_nsbr(bmRes, times, form, len_sbr)
    # the result list of BugReport's issue_id
    issue_id_list = [temp.bug_id for temp in bmRes]
    return issue_id_list

# filter nsbr by different form using bm25f rank model
def bm25F_filter_nsbr(bmRes, times, form, sbrs_num):
    if form == 'ms':
        result = bmRes[:sbrs_num * times]
    elif form == 'rs':
        result_set = set()
        while (len(result_set) < sbrs_num * times):
            result_set.add(roulette_wheel_select(bmRes))
        result = list(result_set)
    elif form=='ns':
        result=bmRes
    else:
        result = []
    return result

# Basic roulette wheel selection: O(N)
def roulette_wheel_select(middleResult):
    '''
    Input: a list of N fitness values (list or tuple)
    Output: selected index
    '''
    if type(middleResult[-1])== np.ndarray:
        sumSim = sum([(middleResult[-1] - x) for x in middleResult])
        # generate a random number
        rndPoint = random.uniform(0, sumSim)
        # calculate the index: O(N)
        accumulator = 0.0
        for i,val in enumerate(middleResult):
            accumulator += (middleResult[-1] - val)
            if accumulator >= rndPoint:
                return i
    else:
        sumREP = sum([(middleResult[-1].REP - x.REP) for x in middleResult])
        # generate a random number
        rndPoint = random.uniform(0, sumREP)
        # calculate the index: O(N)
        accumulator = 0.0
        for val in middleResult:
            accumulator += (middleResult[-1].REP - val.REP)
            if accumulator >= rndPoint:
                return val

if __name__ == '__main__':
    print('Starting data processing..')
    project = 'ambari'
    df, df_filter = data_filter(project, 5, form='ms',train_way='wpp',rank_way='VSM')
    print(len(df_filter))

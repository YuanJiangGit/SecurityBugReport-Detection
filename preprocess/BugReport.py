from collections import defaultdict
import re
class BugReport(object):
    word_inBR_unigram=defaultdict(set)
    word_inBR_bigram=defaultdict(set)
    lenSum_unigram = float(0)
    lenDesc_unigram = float(0)
    lenSum_bigram = float(0)
    lenDesc_bigram =float(0)
    dev_comment_times=defaultdict(int)
    dev_commented_times=defaultdict(int)
    dev_fixed=defaultdict(int)
    getDocID=defaultdict(int)
    id_dev_Reopened={}
    dev_id_Reopened=defaultdict(list)
    allBR=[]
    trainBR=[]
    trainBR_temp=[]
    testBR=[]
    dev_SocialScore={}
    dev_ExperienceScore={}
    devCommenList_5=[[],[],[],[],[]]
    devCommenList_10=[[],[],[],[],[]]
    RecallRate=[0]*25
    def __init__(self):
        self.bugID=''
        self.version=''
        self.prod=''
        self.comp=''
        self.sever=''
        self.commit=''
        self.priority=''
        self.summ=''
        self.summ_unigram=[]
        self.desc_unigram=[]
        self.summ_times_unigram=defaultdict(int)
        self.desc_times_unigram=defaultdict(int)
        self.totalWords_unigram=[]
        self.desc=''
        self.summ_bigram=[]
        self.desc_bigram=[]
        self.totalWords_bigram=[]
        self.summ_times_bigram=defaultdict(int)
        self.desc_times_bigram=defaultdict(int)
        self.fixed_files=[]
        self.report_time=''
    def getcomWord_unigram(self):
        comWord_unigram = defaultdict(set)
        for i in range(len(self.totalWords_unigram)):
            bug_ids = BugReport.word_inBR_unigram[self.totalWords_unigram[i]]
            for bug_id in bug_ids:
                comWord_unigram[bug_id].add(self.totalWords_unigram[i])
        return comWord_unigram


    def getcomWord_bigram(self):
        comWord_bigram = defaultdict(set)
        for i in range(len(self.totalWords_bigram)):
            bug_ids = BugReport.word_inBR_bigram[self.totalWords_bigram[i]]
            for bug_id in bug_ids:
                comWord_bigram[bug_id].add(self.totalWords_bigram[i])
        return comWord_bigram

    def hasNonNumbers(self,inputString):
        return not bool(re.search(r'\d', inputString))

    def prioAndVersDife(self, b2, column_name):
        if column_name == 'version':
            if self.hasNonNumbers(self.version) or self.hasNonNumbers(b2.version):
                return 0.5
            else:
                pre_column_list = self.version.split('.')
                last_column_list = b2.version.split('.')
                weight = [10, 1, 0.1]
                different_value = 0
                for i, j, z in zip(pre_column_list, last_column_list, weight):
                    if self.hasNonNumbers(i) or self.hasNonNumbers(j):continue
                    i_value = re.search(r'\d', i)
                    j_value = re.search(r'\d', j)
                    different_value += abs(float(i_value.group(0)) - float(j_value.group(0))) * z
                return float(1 / (1 + different_value))
        elif column_name == 'priority':
            if self.hasNonNumbers(self.priority) or self.hasNonNumbers(b2.priority):
                return 0.5
            else:
                pre_column_value = re.search(r'\d', self.priority)
                last_column_value = re.search(r'\d', b2.priority)
                return 1 / (1 + abs(float(pre_column_value.group(0)) - float(last_column_value.group(0))))
        else:
            return 0
from BugReport import BugReport
from Result import Result
import math
class REP(object):

    weights = {'w1': 0.9, 'w2': 0.2, 'w_unisum': 3,
               'w_unidesc': 1, 'bf_unisum': 0.5, 'bf_unidesc': 1.0, 'k1_uni': 2, 'k3_uni': 0.5, 'w_bisum': 3,
               'w_bidesc': 1, 'bf_bisum': 0.5, 'bf_bidesc': 1.0, 'k1_bi': 2, 'k3_bi': 0.5}

    trainWeight = {'w1': True, 'w2': True, 'w_unisum': True,
                   'w_unidesc': True, 'bf_unisum': True, 'bf_unidesc': True, 'k1_uni': True, 'k3_uni': True,
                   'w_bisum': True,
                   'w_bidesc': True, 'bf_bisum': True, 'bf_bidesc': True, 'k1_bi': True, 'k3_bi': True}

    def getTFD_unigram(self, str, DocID):
        res = 0
        if BugReport.trainBR[DocID].summ_times_unigram[str] > 0:
            res += REP.weights['w_unisum'] * BugReport.trainBR[DocID].summ_times_unigram[str] / (
                    1.0 - REP.weights['bf_unisum'] + REP.weights['bf_unisum'] * len(
                BugReport.trainBR[DocID].summ_unigram) / (
                            BugReport.lenSum_unigram / len(BugReport.trainBR)))
        if BugReport.trainBR[DocID].desc_times_unigram[str] > 0:
            res += REP.weights['w_unidesc'] * BugReport.trainBR[DocID].desc_times_unigram[str] / (
                    1.0 - REP.weights['bf_unidesc'] + REP.weights['bf_unidesc'] * len(
                BugReport.trainBR[DocID].desc_unigram) / (
                            BugReport.lenDesc_unigram / len(BugReport.trainBR)))
        return res

    def getTFD_bigram(self, str, DocID):
        res = 0
        if BugReport.trainBR[DocID].summ_times_bigram[str] > 0:
            res += REP.weights['w_bisum'] * BugReport.trainBR[DocID].summ_times_bigram[str] / (
                    1.0 - REP.weights['bf_bisum'] + REP.weights['bf_bisum'] * len(
                BugReport.trainBR[DocID].summ_bigram) / (
                            BugReport.lenSum_bigram / len(BugReport.trainBR)))
        if BugReport.trainBR[DocID].desc_times_bigram[str] > 0:
            res += REP.weights['w_bidesc'] * BugReport.trainBR[DocID].desc_times_bigram[str] / (
                    1.0 - REP.weights['bf_bidesc'] + REP.weights['bf_bidesc'] * len(
                BugReport.trainBR[DocID].desc_bigram) / (
                            BugReport.lenDesc_bigram / len(BugReport.trainBR)))
        return res

    def getIDF_unigram(self, str):
        nt = len(BugReport.word_inBR_unigram[str])
        if nt == 0:
            return 0
        return math.log(len(BugReport.trainBR) / nt)

    def getIDF_bigram(self, str):
        nt = len(BugReport.word_inBR_bigram[str])
        if nt == 0:
            return 0
        return math.log(len(BugReport.trainBR) / nt)

    def getWQ_unigram(self, str, docID):
        wf1 = [2.98, 0.287]
        TFQ = 0
        t1 = BugReport.testBR[docID].summ_times_unigram[str]
        t2 = BugReport.testBR[docID].desc_times_unigram[str]
        TFQ += wf1[0] * t1
        TFQ += wf1[1] * t2
        return (REP.weights['k3_uni'] + 1) * TFQ / (REP.weights['k3_uni'] + TFQ)

    def getWQ_bigram(self, str, docID):
        wf2 = [2.999, 0.994]
        TFQ = 0
        t1 = BugReport.testBR[docID].summ_times_bigram[str]
        t2 = BugReport.testBR[docID].desc_times_bigram[str]
        TFQ += wf2[0] * t1
        TFQ += wf2[1] * t2
        return (REP.weights['k3_bi'] + 1) * TFQ / (REP.weights['k3_bi'] + TFQ)

    def getBM25F_text(self, tempBR, i):
        bmRes = []
        for index, br in enumerate(BugReport.trainBR):
            bmScore_unigram = 0
            bmScore_bigram = 0
            # 判断list是否为空，可以用len(myList),或者直接写myList，因为空list就是False
            for term in list(set(tempBR.totalWords_unigram).intersection(br.totalWords_unigram)):
                TFD = self.getTFD_unigram(term, index)
                # k_ctl = 2  # controls parameter k1_unigram ,用REP.weights['k1_uni'] 代替
                bmScore_unigram += self.getIDF_unigram(term) * (
                        TFD / (TFD + REP.weights['k1_uni'])) * self.getWQ_unigram(term, i)
            for term in list(set(tempBR.totalWords_bigram).intersection(br.totalWords_bigram)):
                TFD2 = self.getTFD_bigram(term, index)
                # k_ctl = 2  # controls parameter k1_bigram
                bmScore_bigram += self.getIDF_bigram(term) * (
                        TFD2 / (TFD2 + REP.weights['k1_bi'])) * self.getWQ_bigram(term, i)

            res = REP.weights['w1'] * bmScore_unigram
            res += REP.weights['w2'] * bmScore_bigram

            rTemp = Result()
            rTemp.bug_id = BugReport.trainBR[index].bugID
            rTemp.docID = index
            rTemp.REP = res
            bmRes.append(rTemp)
        return bmRes

    def getWQ_unigram_train(self, str, docID):
        # L = 0.001 # REP.weights['k3_uni']
        # w_summ,w_desc  term在查询（这里是测试报告）的summary和desc中占的权重
        wf1 = [2.98, 0.287]
        TFQ = 0
        t1 = BugReport.trainBR[docID].summ_times_unigram[str]
        t2 = BugReport.trainBR[docID].desc_times_unigram[str]
        TFQ += wf1[0] * t1
        TFQ += wf1[1] * t2
        return (REP.weights['k3_uni'] + 1) * TFQ / (REP.weights['k3_uni'] + TFQ)
    def getWQ_bigram_train(self, str, docID):
        # L = 0.001  #REP.weights['k3_bi']
        # w_summ,w_desc
        wf2 = [2.999, 0.994]
        TFQ = 0
        t1 = BugReport.trainBR[docID].summ_times_bigram[str]
        t2 = BugReport.trainBR[docID].desc_times_bigram[str]
        TFQ += wf2[0] * t1
        TFQ += wf2[1] * t2
        return (REP.weights['k3_bi'] + 1) * TFQ / (REP.weights['k3_bi'] + TFQ)



    #定义rnc函数
    def rnc(self,main,rel,irr,weightIndex):
        sim_rel=self.similarityPerWeight(main,rel,weightIndex)
        sim_irr=self.similarityPerWeight(main,irr,weightIndex)

        sim_dif=sim_irr-sim_rel
        u=1+math.exp(sim_dif)
        du=math.exp(sim_dif)*sim_dif

        dx=du*1.0/(u*math.log(2))
        if math.isnan(dx):
            return 0
        return dx

    #设置权重是否可调整
    def fixWeight(self,indexes):
        for w in indexes:
            REP.trainWeight[w]=False
    def unfixWeight(self,indexes):
        for w in indexes:
            REP.trainWeight[w]=True

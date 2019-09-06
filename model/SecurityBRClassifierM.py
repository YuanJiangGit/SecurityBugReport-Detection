'''
author: jiangyuan
data: 2018/5/17
function: predict security bug report and comparison of multiple models
'''

from SecurityBRClassifier import SecurityClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, \
    f1_score
import matplotlib.pyplot as plt
import re,os
from pandas import DataFrame, Series
from sklearn_evaluation import plot, table


class SecurityClassifierM():
    def __init__(self, classifiers):
        exits_classifiers = {'NB': 'naive_bayes', 'KNN': 'knn', 'LR': 'logistic_regression', 'RF': 'random_forest',
                             'DT': 'decision_tree', 'SVM': 'svm', 'SVMCV': 'svmcs', 'GBDT':'gbdt', 'MLP':'multilayer_perceptron'}
        self.models = []
        self.models_name = []
        for classifier in classifiers:
            if classifier in exits_classifiers:
                self.models_name.append(exits_classifiers[classifier])
                self.models.append(SecurityClassifier(classifier))
            else:
                print('%s分类器不存在'%classifier)

    # train model
    def train(self, train_x, train_y):
        for model in self.models:
            model.train(train_x, train_y)

    # predict binary
    def predict_b(self, X_test):
        y_pred_list = []
        for model in self.models:
            y_pred_list.append(model.predict_b(X_test))
        return y_pred_list

    # predict probability
    def predict_p(self, X_test):
        y_pred_score = []
        for model in self.models:
            y_pred_score.append(model.predict_p(X_test))
        return y_pred_score

    # evaluate binary
    def evaluate_b(self, y, y_pred_list):
        columns = ['TN', 'FN', 'TP', 'FP', 'pd', 'pf', 'prec', 'recall', 'f-measure', 'g-measure', 'accuracy']
        df = DataFrame(columns=columns,index=self.models_name)
        for i, y_pred in enumerate(y_pred_list):
            # report_text = classification_report(y, y_pred, target_names=['nsbr', 'sbr'])
            # report_list = re.sub(r'[\n\s]{1,}', ' ', report_text).strip().split(' ')
            # print(report_list)
            # print('Confusion Matrix:')
            conf_matrix = confusion_matrix(y, y_pred)
            TN = conf_matrix.item((0, 0))
            FN = conf_matrix.item((1, 0))
            TP = conf_matrix.item((1, 1))
            FP = conf_matrix.item((0, 1))

            pd = 100 * TP / (TP + FN)
            pf = 100 * FP / (FP + TN)
            g_measure = 2 * pd * (100 - pf) / (pd + 100 - pf)

            # print('precision:%s' % precision_score(y, y_pred, average='binary'))
            # print('recall:%s' % recall_score(y, y_pred, average='binary'))
            # print('f-measure:%s' % f1_score(y, y_pred, average='binary'))
            prec = 100 * precision_score(y, y_pred, average='binary')
            recall = 100 * recall_score(y, y_pred, average='binary')
            f_measure = 100 * f1_score(y, y_pred, average='binary')

            accuracy = 100*accuracy_score(y, y_pred)
            dict = {'TN': TN, 'FN': FN, 'TP': TP, 'FP': FP, 'pd': pd, 'pf': pf, 'prec': prec,
                    'recall': recall, 'f-measure': f_measure, 'g-measure': g_measure, 'accuracy': accuracy}
            df.loc[self.models_name[i]] = Series(dict)
        return df

    # evaluate probability
    def evaluate_p(self, y, y_score_list,path):
        fig, ax = plt.subplots()
        for y_score in y_score_list:
            plot.roc(y, y_score,ax=ax)

        handles, labels=ax.get_legend_handles_labels()
        # plt.setp(handles,linewidth=1.0)
        ax.legend(handles=handles,labels=self.models_name,loc="lower right")
        fig.savefig(path+'_roc.png')
        plt.show()

        fig, ax = plt.subplots()
        for y_score in y_score_list:
            plot.precision_recall(y, y_score,ax=ax)
        ax.legend(self.models_name,loc='upper right')
        fig.savefig(path + 'precision_recall.png')
        plt.show()
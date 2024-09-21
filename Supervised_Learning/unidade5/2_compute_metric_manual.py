from src.utils import process_diabetes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np


class Metrics:

    def __init__(self, y_pred, y_test):
        self.vp_c1 = 0
        self.vn_c1 = 0
        self.fp_c1 = 0
        self.fn_c1 = 0
        self.vp_c0 = 0
        self.vn_c0 = 0
        self.fp_c0 = 0
        self.fn_c0 = 0

    def set_param_classe1(self):

        for yp, yt in zip(y_pred, y_test):
            if yp == 1 and yt == 1:
                self.vp_c1 = self.vp_c1 + 1
            if yp == 1 and yt == 0:
                self.fp_c1 = self.fp_c1 + 1
            if yp == 0 and yt == 1:
                self.fn_c1 = self.fn_c1 + 1
            if yp == 0 and yt == 0:
                self.vn_c1 = self.vn_c1 + 1

    def set_param_classe2(self):

        for yp, yt in zip(y_pred, y_test):
            if yp == 0 and yt == 0:
                self.vp_c0 = self.vp_c0 + 1
            if yp == 0 and yt == 1:
                self.fp_c0 = self.fp_c0 + 1
            if yp == 1 and yt == 0:
                self.fn_c0 = self.fn_c0 + 1
            if yp == 1 and yt == 1:
                self.vn_c0 = self.vn_c0 + 1


    def compute_acuraccy(self):
        assert len(y_pred) == len(y_test), "Verifica se o número de predições é igual ao número de rótulos verdadeiros."
        num_acertos = np.sum(y_pred == y_test)
        total_amostras = len(y_pred)
        acuracia = (num_acertos / total_amostras) * 100
        return acuracia

    def compute_recall_c1(self):
        recall_c1 = np.sum(self.vp_c1) / (np.sum(self.vp_c1 ) + np.sum(self.fn_c1))
        return recall_c1

    def compute_recall_c0(self):
        recall_c0 = np.sum(self.vp_c0) / (np.sum(self.vp_c0) + np.sum(self.fn_c0))
        return recall_c0

    def compute_precision_c1(self):
        precision_c1 = np.sum(self.vp_c1) / (np.sum(self.vp_c1) + np.sum(self.fp_c1))
        return precision_c1

    def compute_precision_c0(self):
        precision_c0 = np.sum(self.vp_c0)/ (np.sum(self.vp_c0) + np.sum(self.fp_c0))
        return precision_c0

    def compute_f1_score_c1(self):
        recall_c1 = np.sum(self.vp_c1) / (np.sum(self.vp_c1) + np.sum(self.fn_c1))
        precision_c1 = np.sum(self.vp_c1) / (np.sum(self.vp_c1) + np.sum(self.fp_c1))
        if precision_c1 > 0 or recall_c1 > 0:
            f1_score_c1 = 2 * (precision_c1 * recall_c1) / (precision_c1 + recall_c1)
        else:
            f1_score_c1 = 0
        return f1_score_c1
    def compute_f1_score_c0(self):
        recall_c0 = np.sum(self.vp_c0) / (np.sum(self.vp_c0) + np.sum(self.fn_c0))
        precision_c0 = np.sum(self.vp_c0)/ (np.sum(self.vp_c0) + np.sum(self.fp_c0))
        if precision_c0 > 0 or recall_c0 > 0:
            f1_score_c0 = 2 * (precision_c0 * recall_c0) / (precision_c0 + recall_c0)
        else:
            f1_score_c0 = 0
        return f1_score_c0

    def compute_confusion_matriz(self):
        confusion_matrix = [[self.vp_c0, self.fp_c0],
                            [self.fp_c1, self.vp_c1]]
        return confusion_matrix


y_pred, y_test = process_diabetes()
mt = Metrics(y_pred, y_test)
mt.set_param_classe1()
mt.set_param_classe2()

print("acurácia geral:", mt.compute_acuraccy())
#
print("recall classe 0: ", mt.compute_recall_c0())
#
print("recall classe 1: ", mt.compute_recall_c1())
#
print("precision classe 0: ", mt.compute_precision_c0())
#
print("precision classe 1:", mt.compute_precision_c1())
#
print("F1-score classe 0:", mt.compute_f1_score_c0())
#
print("F1-score classe 1:", mt.compute_f1_score_c1())
#
print("Matriz de confusão", mt.compute_confusion_matriz())
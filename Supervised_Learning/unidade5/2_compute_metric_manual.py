import numpy as np

from src.utils import process_diabetes

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
            if yp == 0 and yt == 0:
                self.vn_c1 = self.vn_c1 + 1

    def sum_c1_c0(self):
        vp = self.vp_c1 + self.vp_c0
        vn = self.vn_c1 + self.vn_c0
        fp = self.fp_c1 + self.fp_c0
        fn = self.fn_c1 + self.fn_c0
        return vp, vn, fp, fn
    def compute_acuraccy(self):
        vp, vn, fp, fn = self.sum_c1_c0()
        accuraccy = (vp + vn) /(vp + fn + vn + fp)
        return accuraccy

    def compute_recall_c1(self):

        recall = self.vp_c1 / (self.vp_c1 + self.fn_c1)
        return recall

    def compute_recall_c0(self):
        recall = self.vp_c0 / (self.vp_c0 + self.fn_c0)
        return recall

    def compute_precision_c1(self):
        precision = self.vp_c1 / (self.vp_c1 + self.fp_c1)
        return precision

    def compute_precision_c0(self):
        precision = self.vp_c0 / (self.vp_c0 + self.fp_c0)
        return precision

    def compute_f1_score_c1(self):
        f1_score = 2 * (self.compute_precision_c1() * self.compute_recall_c1()) / (
                    self.compute_precision_c1() + self.compute_recall_c1())
        return f1_score

    def compute_f1_score_c0(self):
        f1_score = 2 * (self.compute_precision_c0() * self.compute_recall_c0()) / (
                    self.compute_precision_c0() + self.compute_recall_c0())
        return f1_score

    def compute_confusion_matriz(self):
        vp, vn,fp,fn = self.sum_c1_c0()
        confusion_matriz = [[vp, vn], [fp, fn]]
        return confusion_matriz


y_pred, y_test = process_diabetes()
mt = Metrics(y_pred, y_test)
mt.set_param_classe1()
mt.set_param_classe2()

print("acurácia geral:", mt.compute_acuraccy())
#
print(f"recall classe 0: {mt.compute_recall_c0()}")
#
print(f"recall classe 1: {mt.compute_recall_c1()}")
#
print(f"precision classe 0: {mt.compute_precision_c0()}")
#
print(f"precision classe 1: {mt.compute_precision_c1()}")
#
print(f"F1-score classe 0: {mt.compute_f1_score_c0()}")
#
print(f"F1-score classe 1: {mt.compute_f1_score_c1()}")
#
print(f"Matriz de confusão: {mt.compute_confusion_matriz()}")
#
from src.utils import process_diabetes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def compute_acuraccy(y_pred ,y_test):
    assert len(y_pred) == len(y_test), "Verifica se o número de predições é igual ao número de rótulos verdadeiros."
    num_acertos = np.sum(y_pred == y_test)
    total_amostras = len(y_pred)
    acuracia = (num_acertos / total_amostras) * 100
    return acuracia


def compute_recall(y_pred , y_test):
    assert len(y_pred) == len(y_test)
    verdadeiros_positivos = np.sum(np.logical_and(y_pred == 1, y_test == 1))
    falsos_negativos = np.sum(np.logical_and(y_pred == 0, y_test == 1))
    recall = verdadeiros_positivos / (verdadeiros_positivos + falsos_negativos)
    return recall

def compute_precision(y_pred, y_test):
    assert len(y_pred) == len(y_test)
    verdadeiros_positivos = np.sum(np.logical_and(y_pred == 1, y_test == 1))
    falsos_positivos = np.sum(np.logical_and(y_pred == 1, y_test == 0))

    precision = verdadeiros_positivos / (verdadeiros_positivos + falsos_positivos)
    return precision


def compute_f1_score(y_pred,y_test):
    assert len(y_pred) == len(y_test)

    verdadeiros_positivos = np.sum(np.logical_and(y_pred == 1, y_test == 1))
    falsos_positivos = np.sum(np.logical_and(y_pred == 1, y_test == 0))
    falsos_negativos = np.sum(np.logical_and(y_pred == 0, y_test == 1))

    precision = compute_precision(y_pred, y_test)
    recall = compute_recall(y_pred,y_test)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def compute_confusion_matriz(y_pred,y_test):
    assert len(y_pred) == len(y_test)

    verdadeiros_positivos = np.sum(np.logical_and(y_pred == 1, y_test == 1))
    falsos_positivos = np.sum(np.logical_and(y_pred == 1, y_test == 0))
    falsos_negativos = np.sum(np.logical_and(y_pred == 0, y_test == 1))
    verdadeiros_negativos = np.sum(np.logical_and(y_pred == 0, y_test == 0))

    confusion_matrix = np.array([[verdadeiros_negativos, falsos_positivos],
                                 [falsos_negativos, verdadeiros_positivos]])

    return confusion_matrix

y_pred ,y_test = process_diabetes()



print("acurácia geral: {}".format(compute_acuraccy(y_pred,y_test)))
#
print("recall classe 0: {} ".format(compute_recall(y_pred,y_test == 0)))
#
print("recall classe 1: {}".format(compute_recall(y_pred,y_test == 1)))
#
print("precision classe 0: {} ".format(compute_precision(y_pred,y_test == 0)))
#
print("precision classe 1: {}".format(compute_precision(y_pred,y_test == 1)))
#
print("F1-score classe 0: {}".format(compute_f1_score(y_pred,y_test == 0)))
#
print("F1-score classe 1: {}".format(compute_f1_score(y_pred,y_test == 1)))
#
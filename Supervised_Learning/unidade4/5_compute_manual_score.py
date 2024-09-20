import numpy as np
from src.utils import processing_all_features_sales_clean


def compute_RSS(predictions, y): # Calcula a soma dos quadrados.
    RSS = np.sum((y - predictions) ** 2)
    return RSS


def compute_MSE(predictions, y): # Calcula a média dos quadrados.
    MSE = np.mean((y - predictions) ** 2)
    return MSE


def compute_RMSE(predictions, y): # Calcula a raíz quadrada do MSE.
    RMSE = np.sqrt(compute_MSE(predictions, y))
    return RMSE


def compute_R_squared(predictions, y): # Calcula o R².
    ss_res = np.sum((y - predictions) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

# Chama a função para carregar as features.
X, y, predictions = processing_all_features_sales_clean()

# Impressão dos resultados.
print("RSS: {}".format(compute_RSS(predictions, y)))
print("MSE: {}".format(compute_MSE(predictions, y)))
print("RMSE: {}".format(compute_RMSE(predictions, y)))
print("R^2: {}".format(compute_R_squared(predictions, y)))

import numpy as np
from src.utils import processing_all_features_sales_clean

def compute_RSS(predictions, y):
    # RSS é a soma dos quadrados das diferenças entre predições e valores reais
    residuals = y - predictions
    RSS = np.sum(residuals**2)
    return RSS

def compute_MSE(predictions, y):
    # MSE é a média dos quadrados dos resíduos
    RSS = compute_RSS(predictions, y)
    MSE = RSS / len(y)
    return MSE

def compute_RMSE(predictions, y):
    # RMSE é a raiz quadrada do MSE
    MSE = compute_MSE(predictions, y)
    RMSE = np.sqrt(MSE)
    return RMSE

def compute_R_squared(predictions, y):
    # RSS e TSS são necessários para calcular o R^2
    RSS = compute_RSS(predictions, y)
    TSS = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (RSS / TSS)
    return r_squared

# Supondo que a função processing_all_features_sales_clean retorna X, y e predições
X, y, predictions = processing_all_features_sales_clean()

print("RSS: {}".format(compute_RSS(predictions, y)))
print("MSE: {}".format(compute_MSE(predictions, y)))
print("RMSE: {}".format(compute_RMSE(predictions, y)))
print("R^2: {}".format(compute_R_squared(predictions, y)))

import numpy as np
from src.utils import processing_all_features_sales_clean

def compute_RSS(predictions,y):
    RSS = np.sum((y - predictions) ** 2)
    return RSS
def compute_MSE(predictions,y):
    MSE = np.mean((y - predictions) ** 2)
    return MSE
def compute_RMSE(predictions,y):
    RMSE = np.sqrt(compute_MSE(predictions, y))
    return RMSE
def compute_R_squared(predictions,y):
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = compute_RSS(predictions, y)
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared


X,y,predictions = processing_all_features_sales_clean()


print("RSS: {}".format(compute_RSS(predictions,y)))
print("MSE: {}".format(compute_MSE(predictions,y)))
print("RMSE: {}".format(compute_RMSE(predictions,y)))
print("R^2: {}".format(compute_R_squared(predictions,y)))
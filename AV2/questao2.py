from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Implemente uma regressão linear utilizando somente este atributo mais relevante, para predição do atributo alvo determinado na questão 1
emissao_new = pd.read_csv("emissao_new.csv")
y = emissao_new["emissions_tons"].values
X = emissao_new["year"].values.reshape(-1, 1)

# Create the model
reg = LinearRegression()
# Fit the model to the data
reg.fit(X, y)
# Make predictions
predictions = reg.predict(X)
print(predictions[:5])
# Create scatter plot
plt.scatter(X, y, color="red")
# Create line plot
plt.plot(X, predictions, color="black")
plt.xlabel("Year")
plt.ylabel("Thailand CO2 Emission")

# Display the plot
plt.show()

#Determine também os valores: RSS, MSE, RMSE e R_squared para esta regressão baseada somente no atributo mais relevante
def compute_RSS(predictions,y):
    RSS = np.sum(np.square(y - predictions))
    return RSS

def compute_MSE(predictions,y):
    MSE= np.sum(np.square(y-predictions))/len(predictions)
    return MSE
def compute_RMSE(predictions,y):
    MSE = compute_MSE(predictions, y)
    RMSE = np.sqrt(MSE)
    return RMSE
def compute_R_squared(predictions,y):
    var_pred = np.sum(np.square(predictions - np.mean(y)))
    var_data = np.sum(np.square(y-np.mean(y)))
    r_squared = np.divide(var_pred, var_data)
    return r_squared
print("RSS: {}".format(compute_RSS(predictions,y)))
print("MSE: {}".format(compute_MSE(predictions,y)))
print("RMSE: {}".format(compute_RMSE(predictions,y)))
print("R^2: {}".format(compute_R_squared(predictions,y)))
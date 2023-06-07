import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
emissao_df = pd.read_csv('dados.csv')
print(emissao_df)


X = emissao_df[['Fuel Consumption Hwy (L/100 km)']]
y = emissao_df['CO2 Emissions(g/km)']


regressor = LinearRegression()

# Treinar o modelo utilizando o atributo mais relevante
regressor.fit(X, y)
pred = regressor.predict(X)

# Exibir as predições
print(pred)



#metricas
def compute_RSS(predictions,y):
    sub_squared= np.square(y - predictions)
    RSS = np.sum(sub_squared)
    return RSS

def compute_MSE(predictions,y):
    RSS= compute_RSS(predictions, y)
    MSE= np.divide(RSS, len(predictions))
    return MSE

def compute_RMSE(predictions,y):
    MSE= compute_MSE(predictions, y)
    RMSE = np.sqrt(MSE)
    return RMSE

def compute_R_squared(predictions,y):
    var_pred = np.sum(np.square(predictions - np.mean(y)))
    var_data = np.sum(np.square(y - np.mean(y)))
    r_squared = np.divide(var_pred, var_data)
    return r_squared

print("RSS: {}".format(compute_RSS(pred, y)))
print("MSE: {}".format(compute_MSE(pred, y)))
print("RMSE: {}".format(compute_RMSE(pred, y)))
print("R^2: {}".format(compute_R_squared(pred, y)))


# Plotar a nuvem de pontos do atributo e a reta de regressão
plt.scatter(X, y, color='red', label='Dados')
plt.plot(X, pred, color='black', linewidth=2, label='Regressão Linear')
plt.xlabel('Fuel Consumption Hwy (L/100 km)')
plt.ylabel('CO2 Emissions(g/km)')
plt.legend()
plt.show()
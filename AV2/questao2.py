'''Utilizando o atributo mais relevante calculado na questão 1, implemente uma regressão linear
utilizando somente este atributo mais relevante, para predição do atributo alvo determinado na questão
1 também. Mostre o gráfico da reta de regressão em conjunto com a nuvem de atributo. Determine também
os valores: RSS, MSE, RMSE e R_squared para esta regressão baseada somente no atributo mais relevante.
Obs: Registrar na seção de resultados a análise realizada e discutir sobre os resultados encontrados.'''

# Importando as bibliotecas necessárias
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


car_price = pd.read_csv("car_price_atualizado.csv")
pd.set_option('display.max_columns', None)


X = car_price["make_ed"].values.reshape(-1, 1) # marca
y = car_price["price"].values # preço

# Criando o modelo
reg = LinearRegression()

# Ajustar o modelo aos dados
reg.fit(X, y)

# Fazendo previsões
previsoes = reg.predict(X)

print("\nPrevisões da Regressão Linear:")
print(previsoes[:5])

# Criando o gráfico
plt.scatter(X, y, color="pink")
plt.plot(X, previsoes, color="black")
plt.xlabel("Marca")
plt.ylabel("Preço")
plt.show()


def compute_RSS(previsoes,y):
    RSS = np.sum(np.square(y - previsoes))
    return RSS

def compute_MSE(previsoes,y):
    MSE= np.sum(np.square(y-previsoes))/len(previsoes)
    return MSE

def compute_RMSE(previsoes,y):
    MSE = compute_MSE(previsoes, y)
    RMSE = np.sqrt(MSE)
    return RMSE

def compute_R_squared(previsoes,y):
    var_pred = np.sum(np.square(previsoes - np.mean(y)))
    var_data = np.sum(np.square(y-np.mean(y)))
    r_squared = np.divide(var_pred, var_data)
    return r_squared

print("\nRSS: {}".format(compute_RSS(previsoes,y)))
print("\nMSE: {}".format(compute_MSE(previsoes,y)))
print("\nRMSE: {}".format(compute_RMSE(previsoes,y)))
print("\nR^2: {}".format(compute_R_squared(previsoes,y)))




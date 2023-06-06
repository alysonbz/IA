#bibliotecas
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# Carregar o dataset Ev
Ev = pd.read_csv('dados.csv')

# Carregar o dataset dados
dados = pd.read_csv('car_price_prediction.csv')

# Atributo mais relevante
X = Ev.values
y = dados['Price'].values

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciar o modelo de regressão linear
regressor = LinearRegression()

# Ajustar o modelo aos dados de treinamento
regressor.fit(X_train, y_train)

# Fazer previsões nos dados de teste
y_pred = regressor.predict(X_test)

# Plotar o gráfico da nuvem de pontos e da reta de regressão
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Observações')
plt.plot(X_test, y_pred, color='red', label='Reta de Regressão')
plt.xlabel('Engine volume')
plt.ylabel('Price')
plt.title('Regressão Linear - Engine volume vs Price')
plt.legend()
plt.show()











# Função para calcular o RSS (Residual Sum of Squares)
def compute_RSS(predictions, y):
    sub_squared = np.square(y - predictions)
    RSS = np.sum(sub_squared)
    return RSS

# Função para calcular o MSE (Mean Squared Error)
def compute_MSE(predictions, y):
    RSS = compute_RSS(predictions, y)
    MSE = np.divide(RSS, len(predictions))
    return MSE

# Função para calcular o RMSE (Root Mean Squared Error)
def compute_RMSE(predictions, y):
    MSE = compute_MSE(predictions, y)
    RMSE = np.sqrt(MSE)
    return RMSE

# Função para calcular o R² (R-squared)
def compute_R_squared(predictions, y):
    var_pred = np.sum(np.square(predictions - np.mean(y)))
    var_data = np.sum(np.square(y - np.mean(y)))
    r_squared = np.divide(var_pred, var_data)
    return r_squared

# Calcular as métricas manualmente
rss = compute_RSS(y_pred, y_test)
mse = compute_MSE(y_pred, y_test)
rmse = compute_RMSE(y_pred, y_test)
r_squared = compute_R_squared(y_pred, y_test)

print("RSS:", rss)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r_squared)

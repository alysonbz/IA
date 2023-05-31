import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Carregar o dataset atualizado
data = pd.read_csv("forest_fires_updated.csv")

# Extrair o atributo mais relevante (calculado na questão 1)
X = data["month"].values.reshape(-1, 1)
y = data["area"].values


# Criar o modelo de regressão linear
model = LinearRegression()

# Ajustar o modelo aos dados
model.fit(X, y)

# Realizar previsões
y_pred = model.predict(X)

# Calcular as métricas de avaliação
rss = np.sum((y_pred - y) ** 2)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r_squared = model.score(X, y)

# Imprimir os valores das métricas
print("RSS:", rss)
print("MSE:", mse)
print("RMSE:", rmse)
print("R^2:", r_squared)

# Plotar o gráfico da reta de regressão e a nuvem de pontos
plt.scatter(X, y, color='b', label='Observations')
plt.plot(X, y_pred, color='r', linewidth=2, label='Linear Regression')
plt.xlabel("Month")
plt.ylabel("Area")
plt.title("Linear Regression: Month vs Area")
plt.legend()
plt.show()
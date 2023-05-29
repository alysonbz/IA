import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Carregar o dataset
df = pd.read_csv("df.csv")
# Selecionar o atributo alvo e o atributo mais relevante
target_attribute = 'BodyFat'
relevant_attribute = 'Weight'
# Extrair os valores do atributo alvo e do atributo mais relevante
X = df[relevant_attribute].values.reshape(-1, 1)
y = df[target_attribute].values
# Criar uma instância do modelo de regressão linear
regression_model = LinearRegression()
# Treinar o modelo
regression_model.fit(X, y)
# Realizar a previsão usando o atributo mais relevante
y_pred = regression_model.predict(X)
# Calcular as métricas de avaliação
rss = np.sum((y - y_pred) ** 2)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)
# Plotar a reta de regressão e a nuvem de pontos
plt.scatter(X, y, color='blue', label='Dados reais')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regressão Linear')
plt.xlabel(relevant_attribute)
plt.ylabel(target_attribute)
plt.legend()
plt.title('Regressão Linear com Atributo mais Relevante')
plt.show()
# Imprimir os resultados
print("RSS:", rss)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)
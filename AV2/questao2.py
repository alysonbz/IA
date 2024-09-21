"""
Utilizando o atributo mais relevante calculado na questão 1,implemente uma regressão linear
utilizando somente este atributo mais relevante, para predição do atributo alvo determinado na
questão 1 também. Mostre o gráfico da reta de regressão em conjunto com a nuvem de atributo.
Determine também os valores: RSS, MSE, RMSE e R_squared para esta regressão baseada somente no atributo
mais relevante. Obs: Registrar na seção de resultados a análise realizada e discutir sobre os resultados encontrados.
"""
from src.utils import load_laptopPrice_dataset_cleaned
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

laptop = load_laptopPrice_dataset_cleaned()

# Definindo o atributo mais relevante e o alvo
X = laptop[['graphic_card_gb']].values  # Variável independente
y = laptop['Price'].values  # Variável dependente

# Criando o modelo de Regressão Linear
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Prevendo os valores com base no modelo
y_pred = linear_regressor.predict(X)

# Cálculo dos indicadores
RSS = np.sum((y - y_pred) ** 2)
MSE = mean_squared_error(y, y_pred)
RMSE = np.sqrt(MSE)
R_squared = r2_score(y, y_pred)

# Exibindo os resultados
print(f"RSS: {RSS}")
print(f"MSE: {MSE}")
print(f"RMSE: {RMSE}")
print(f"R_squared: {R_squared}")

# Plotando o gráfico da regressão linear
plt.scatter(X, y, color='blue', label='Dados reais')
plt.plot(X, y_pred, color='red', label='Linha de Regressão')
plt.title('Regressão Linear - Price vs Graphic Card GB')
plt.xlabel('Graphic Card GB')
plt.ylabel('Price')
plt.legend()
plt.show()

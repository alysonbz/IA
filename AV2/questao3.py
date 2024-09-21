"""
Utilizando o atributo mais relevante calculado na questão 1, implemente uma regressão
linear utilizando somente este atributo mais relevante, para predição do atributo alvo
determinado na questão 1 também. Mostre o gráfico da reta de regressão em conjunto com
a nuvem de atributo. Determine também os valores: RSS, MSE, RMSE e R_squared para esta
regressão baseada somente no atributo mais relevante. Obs: Registrar na seção de resultados
a análise realizada e discutir sobre os resultados encontrados.
"""
from src.utils import load_laptopPrice_dataset_cleaned
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


laptop = load_laptopPrice_dataset_cleaned()
# Selecionar apenas o atributo mais relevante
X = laptop[['graphic_card_gb']]
y = laptop['Price']

# 1. Criar e treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X, y)

# 2. Fazer previsões
y_pred = model.predict(X)

# 3. Calcular métricas
rss = np.sum((y - y_pred) ** 2)  # Residual Sum of Squares
mse = mean_squared_error(y, y_pred)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
r_squared = r2_score(y, y_pred)  # R-squared

# 4. Mostrar resultados
print(f"RSS: {rss}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r_squared}")

# 5. Plotar a reta de regressão
plt.scatter(X, y, color='blue', label='Dados reais')
plt.plot(X, y_pred, color='red', label='Reta de regressão')
plt.title('Regressão Linear: Previsão de Preços de Laptops')
plt.xlabel('Graphic Card (GB)')
plt.ylabel('Price')
plt.legend()
plt.show()
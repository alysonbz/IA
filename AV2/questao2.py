"""
- Questão 2
    Utilizando o atributo mais relevante calculado na questão 1, implemente uma regressão linear utilizando somente este atributo mais
    relevante, para predição do atributo alvo determinado na questão 1 também. Mostre o gráfico da reta de regressão  em conjunto com a nuvem
    de atributo.

    Determine também os valores:
    RSS, MSE, RMSE e R_squared para esta regressão baseada somente no atributo mais relevante.

    Obs: Registrar na seção de resultados a análise realizada e discutir sobre os resultados encontrados.
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.utils import load_activision_blizzard_dataset

act_blz = load_activision_blizzard_dataset()

# Definir o atributo mais relevante (Adj Close) e o alvo (Close)
X = act_blz[['Adj Close']].values
y = act_blz['Close'].values

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o modelo de regressão linear
lr = LinearRegression()

# Treinar o modelo
lr.fit(X_train, y_train)

# Fazer previsões
y_pred = lr.predict(X_test)

# Calcular as métricas
RSS = np.sum(np.square(y_test - y_pred))
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
R_squared = r2_score(y_test, y_pred)

print("RSS:", RSS)
print("MSE:", MSE)
print("RMSE:", RMSE)
print("R²:", R_squared)

# Montando a reta de regressão junto com os pontos de dados
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Dados Reais')
plt.plot(X_test, y_pred, color='red', label='Reta da Regressão')
plt.xlabel('Adj Close')
plt.ylabel('Close')
plt.title('Regressão Linear: Adj Close vs Close')
plt.legend()
plt.show()
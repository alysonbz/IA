"""
### Questao2.py```

Utilizando o atributo mais relevante calculado na questão 1, implemente uma regressão linear utilizando somente este atributo mais
relevante, para predição do atributo alvo determinado na questão 1 também. Mostre o gráfico da reta de regressão  em conjunto com a nuvem
de atributo.
Determine também os valores:
RSS, MSE, RMSE e R_squared para esta regressão baseada somente no atributo mais relevante.
Obs: Registrar na seção de resultados a análise realizada e discutir sobre os resultados encontrados.

"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from src.utils import load_smart_watch_prices_dataset

# Carregar o dataset usando a função de utilidade
swp = load_smart_watch_prices_dataset()

# Remover o símbolo de dólar e converter 'Price (USD)' para numérico
swp['Price (USD)'] = swp['Price (USD)'].replace({r'\$': '', ',': ''}, regex=True).astype(float)

# Substituir valores como 'Unlimited' por um valor alto (999) e remover 'hours' e 'days'
swp['Battery Life (days)'] = swp['Battery Life (days)'].replace({'hours': '', 'days': '', 'Unlimited': 999}, regex=True)

# Converter a coluna 'Battery Life (days)' para numérico
swp['Battery Life (days)'] = pd.to_numeric(swp['Battery Life (days)'], errors='coerce')

# Definir o atributo mais relevante ('Battery Life (days)') e o alvo ('Price (USD)')
X = swp[['Battery Life (days)']].values
y = swp['Price (USD)'].values

# Remover valores NaN antes de dividir os dados
mask = ~np.isnan(X).ravel() & ~np.isnan(y)
X = X[mask].reshape(-1, 1)
y = y[mask]

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o modelo de regressão linear
model = LinearRegression()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Calcular as métricas
rss = np.sum(np.square(y_test - y_pred))
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(y_test, y_pred)

# Exibir os resultados
print(f"RSS: {rss}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r_squared}")

# Plotar a reta de regressão junto com os pontos de dados
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Dados Reais')
plt.plot(X_test, y_pred, color='red', label='Reta de Regressão')
plt.xlabel('Battery Life (days)')
plt.ylabel('Price (USD)')
plt.title('Regressão Linear: Battery Life (days) vs Price (USD)')
plt.legend()
plt.show()

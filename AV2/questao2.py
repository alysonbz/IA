import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from src.utils import load_lenovo_share_prices  # Importar a função de utils.py

# 1. Carregar o dataset usando a função do arquivo utils.py
lenovo_df = load_lenovo_share_prices()

# 2. Remover linhas com valores ausentes
lenovo_df = lenovo_df.dropna(subset=['High', 'Close'])

# 3. Selecionar o atributo mais relevante (High) e o alvo (Close)
X = lenovo_df[['High']]  # Atributo mais relevante
y = lenovo_df['Close']   # Atributo alvo

# 4. Criar e treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X, y)

# 5. Fazer previsões
y_pred = model.predict(X)

# 6. Calcular RSS, MSE, RMSE e R²
RSS = ((y - y_pred) ** 2).sum()
MSE = mean_squared_error(y, y_pred)
RMSE = MSE ** 0.5
R_squared = r2_score(y, y_pred)

# Exibir os resultados
print(f"RSS: {RSS}")
print(f"MSE: {MSE}")
print(f"RMSE: {RMSE}")
print(f"R-squared: {R_squared}")

# 7. Gráfico da reta de regressão com nuvem de pontos
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label="Dados Reais")
plt.plot(X, y_pred, color='red', label="Reta de Regressão")
plt.title("Regressão Linear - Previsão do Preço de Fechamento")
plt.xlabel("Preço Mais Alto do Dia (High)")
plt.ylabel("Preço de Fechamento (Close)")
plt.legend()
plt.grid(True)
plt.show()

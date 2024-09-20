from src.utils import load_ferrari_dataset
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Lendo o dataset.
ferrari = load_ferrari_dataset()

# Definir os atributos mais relevantes (Close) e o alvo (Adj Close).
X = ferrari[['Close']].values
y = ferrari['Adj Close'].values

# Dividir os dados em conjuntos de treino e teste.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o modelo de regressão linear.
model = LinearRegression()

# Treinar o modelo.
model.fit(X_train, y_train)

# Fazer previsões.
y_pred = model.predict(X_test)

# Calcular as métricas.
rss = np.sum(np.square(y_test - y_pred))
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(y_test, y_pred)

# Mostrar os valores de RSS, MSE, RMSE & R².
print("RSS:", rss)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r_squared)

# Plotar a reta de regressão junto com os pontos de dados.
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Dados Reais')
plt.plot(X_test, y_pred, color='red', label='Reta de Regressão')
plt.xlabel('Close Price')
plt.ylabel('Adj Close Price')
plt.title('Regressão Linear: Close Price vs Adj Close Price')
plt.legend()
plt.show()
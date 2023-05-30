#importe as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Carregar o dataset
data = pd.read_csv(r'C:\Users\eryka\Downloads\archive\Samsung Electronics.csv')

# Separar atributo alvo e atributo relevante
X = data[['Close']]
y = data[['Close']]

from sklearn.model_selection import train_test_split

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar objeto do modelo de regressão linear
model = LinearRegression()

# Ajustar o modelo aos dados de treinamento
model.fit(X_train, y_train)

# Fazer previsões com base nos dados de teste
y_pred = model.predict(X_test)

# Calcular o RSS (Residual Sum of Squares)
rss = np.sum((y_test - y_pred) ** 2)

# Calcular o MSE (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)

# Calcular o RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)

# Calcular o R-squared (Coefficient of Determination)
r2 = r2_score(y_test, y_pred)

# Plot da reta de regressão e nuvem de pontos
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Dados de teste')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Reta de regressão')
plt.xlabel('Close')
plt.ylabel('Close')
plt.title('Regressão Linear - Previsão do preço de fechamento das ações da Samsung')
plt.legend()
plt.show()


# Imprimir os resultados
print("RSS:", rss)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)


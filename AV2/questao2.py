# Importando as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Carregar dataset
file_path = 'C:\\Users\\Neto\\Downloads\\IA\\AV2\\Dataset\\Sample - Superstore.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Seleção dos atributos
X = df[['Profit']]  # Atributo mais relevante (Profit)
y = df['Sales']     # Atributo alvo (Sales)

#Treinando o modelo
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

# Plotando a reta de regressão com a nuvem de pontos
plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.5, label='Dados reais')
plt.plot(X, y_pred, color='red', label='Reta de Regressão')
plt.title('Reta de Regressão Linear: Profit vs Sales')
plt.xlabel('Profit')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# Calculo do RSS, MSE, RMSE e R²
RSS = ((y - y_pred) ** 2).sum()
MSE = mean_squared_error(y, y_pred)
RMSE = mean_squared_error(y, y_pred, squared=False)
R_squared = r2_score(y, y_pred)

# Resultados
print(f"Residual Sum of Squares (RSS): {RSS}")
print(f"Mean Squared Error (MSE): {MSE}")
print(f"Root Mean Squared Error (RMSE): {RMSE}")
print(f"R² (R_squared): {R_squared}")

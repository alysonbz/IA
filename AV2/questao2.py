# Importação das bibliotecas necessárias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Carregando o dataset
file_path = 'C:\Users\azjoa\Downloads\thailand_co2_emission_1987_2022'
df = pd.read_csv(file_path)

# Codificação da variável categórica 'fuel_type' (usaremos one-hot encoding)
df_encoded = pd.get_dummies(df, columns=['fuel_type'], drop_first=True)

# Definindo a variável dependente (target) e independente (atributo mais relevante)
X = df_encoded[['fuel_type_coal', 'fuel_type_gas', 'fuel_type_oil']]  # Variáveis dummy criadas para fuel_type
y = df['emissions_tons']

# Dividindo os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Implementação da regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazendo predições com o conjunto de teste
y_pred = model.predict(X_test)

# Gráfico da reta de regressão em conjunto com a nuvem de pontos
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('Valores Reais de Emissões (em toneladas)')
plt.ylabel('Valores Previstos de Emissões (em toneladas)')
plt.title('Gráfico da Reta de Regressão')
plt.show()

# Cálculo de RSS, MSE, RMSE e R_squared
RSS = sum((y_test - y_pred) ** 2)  # Residual Sum of Squares
MSE = mean_squared_error(y_test, y_pred)  # Mean Squared Error
RMSE = MSE ** 0.5  # Root Mean Squared Error
R_squared = r2_score(y_test, y_pred)  # Coeficiente de Determinação

# Exibindo os resultados dos erros
print(f"RSS: {RSS}")
print(f"MSE: {MSE}")
print(f"RMSE: {RMSE}")
print(f"R²: {R_squared}")

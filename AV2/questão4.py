import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from src.utils import load_lenovo_share_prices  # Importar a função de utils.py

# 1. Carregar o dataset
lenovo_df = load_lenovo_share_prices()

# 2. Remover linhas com valores ausentes
lenovo_df = lenovo_df.dropna(subset=['Open', 'High', 'Low', 'Volume', 'Close'])

# 3. Selecionar os atributos relevantes para regressão
X = lenovo_df[['Open', 'High', 'Low', 'Volume']]  # Atributos relevantes
y = lenovo_df['Close']  # Alvo

# 4. Criar o modelo de regressão linear
linear_model = LinearRegression()

# 5. Definir o K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 6. Avaliar o modelo usando Cross-Validation
# Usar MSE como métrica
scores = cross_val_score(linear_model, X, y, scoring='neg_mean_squared_error', cv=kf)

# Calcular o MSE médio e o RMSE (raiz do MSE médio)
mse_mean = -scores.mean()  # Média do MSE (invertido porque o GridSearchCV retorna negativo)
rmse_mean = mse_mean ** 0.5  # RMSE

print(f"MSE médio da Regressão Linear (K-Fold CV): {mse_mean}")
print(f"RMSE médio da Regressão Linear (K-Fold CV): {rmse_mean}")

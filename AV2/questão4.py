# Importação das bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error

# Carregando o dataset
file_path = 'C:\Users\azjoa\Downloads\thailand_co2_emission_1987_2022'
df = pd.read_csv(file_path)

# Codificação da variável categórica 'fuel_type' (usaremos one-hot encoding)
df_encoded = pd.get_dummies(df, columns=['fuel_type', 'source'], drop_first=True)

# Definindo a variável dependente (target) e variáveis independentes
X = df_encoded.drop(columns=['emissions_tons', 'year', 'month'])  # Removendo 'emissions_tons' (alvo) e atributos irrelevantes
y = df['emissions_tons']

# Normalizando os dados (escalando)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Implementação do modelo de Regressão Linear
linear_model = LinearRegression()

# Configuração do kfold (5 folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Avaliação do modelo usando cross-validation
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
mse_scores = cross_val_score(linear_model, X_scaled, y, cv=kf, scoring=mse_scorer)
r2_scores = cross_val_score(linear_model, X_scaled, y, cv=kf, scoring='r2')

# Cálculo da média do MSE e do R²
mean_mse = -mse_scores.mean()  # Convertendo para valores positivos
mean_r2 = r2_scores.mean()

# Exibindo os resultados da Regressão Linear
print(f"Desempenho da Regressão Linear com Cross-Validation (5-fold):")
print(f"MSE médio: {mean_mse}")
print(f"R² médio: {mean_r2}")

# Comparando com os resultados de Lasso e Ridge da Questão 3
print("\nComparação com os regressores da Questão 3:")
print(f"Lasso - Melhor MSE: {best_lasso_score}")
print(f"Lasso - Melhor R²: {lasso_r2}")
print(f"Ridge - Melhor MSE: {best_ridge_score}")
print(f"Ridge - Melhor R²: {ridge_r2}")

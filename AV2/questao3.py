# Importação das bibliotecas necessárias
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Carregando o dataset
file_path = 'C:\Users\azjoa\Downloads\thailand_co2_emission_1987_2022'
df = pd.read_csv(file_path)

# Codificação da variável categórica 'fuel_type' (usaremos one-hot encoding)
df_encoded = pd.get_dummies(df, columns=['fuel_type', 'source'], drop_first=True)

# Definindo a variável dependente (target) e variáveis independentes
X = df_encoded.drop(columns=['emissions_tons', 'year', 'month'])  # Removendo 'emissions_tons' (alvo) e atributos irrelevantes
y = df['emissions_tons']

# Dividindo os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizando os dados (escalando)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definindo os parâmetros para o grid search
param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100, 1000]  # Valores de alpha a serem testados
}

# Implementação do Lasso
lasso = Lasso()
grid_lasso = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_lasso.fit(X_train_scaled, y_train)

# Implementação do Ridge
ridge = Ridge()
grid_ridge = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_ridge.fit(X_train_scaled, y_train)

# Melhor configuração e score para Lasso
best_lasso_alpha = grid_lasso.best_params_['alpha']
best_lasso_score = -grid_lasso.best_score_

# Melhor configuração e score para Ridge
best_ridge_alpha = grid_ridge.best_params_['alpha']
best_ridge_score = -grid_ridge.best_score_

# Print das melhores configurações e scores
print(f"Melhor configuração para Lasso: alpha={best_lasso_alpha}, MSE={best_lasso_score}")
print(f"Melhor configuração para Ridge: alpha={best_ridge_alpha}, MSE={best_ridge_score}")

# Avaliando os modelos no conjunto de teste
lasso_best = Lasso(alpha=best_lasso_alpha)
lasso_best.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_best.predict(X_test_scaled)

ridge_best = Ridge(alpha=best_ridge_alpha)
ridge_best.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_best.predict(X_test_scaled)

# Métricas para o Lasso
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
lasso_r2 = r2_score(y_test, y_pred_lasso)

# Métricas para o Ridge
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_r2 = r2_score(y_test, y_pred_ridge)

# Exibindo as métricas
print(f"Desempenho no conjunto de teste - Lasso: MSE={lasso_mse}, R²={lasso_r2}")
print(f"Desempenho no conjunto de teste - Ridge: MSE={ridge_mse}, R²={ridge_r2}")

# Gráfico dos valores reais vs preditos (Lasso e Ridge)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

# Lasso
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lasso, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.title('Lasso - Valores Reais vs Preditos')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Preditos')

# Ridge
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_ridge, color='green', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.title('Ridge - Valores Reais vs Preditos')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Preditos')

plt.tight_layout()
plt.show()

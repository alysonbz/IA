"""
### Questao4.py```

Utilizando kfold e cross-validation faça uma regressão linear utilizando os mesmos atributos definidos na questão 3.
Obs: Com os resultados obtidos na questão 3 e da questão 4 faça uma comparação entre os desempenhos. Escolha o regressor adequado
e informe o motivo da escolha. Discuta sobre as limitações e acertos encontrados.

"""

import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np
from src.utils import load_smart_watch_prices_dataset

# Carregar o dataset usando a função de utilidade
swp = load_smart_watch_prices_dataset()

# Remover o símbolo de dólar e converter 'Price (USD)' para numérico
swp['Price (USD)'] = swp['Price (USD)'].replace({r'\$': '', ',': ''}, regex=True).astype(float)

# Substituir valores como 'Unlimited' por um valor alto (999) e remover 'hours' e 'days'
swp['Battery Life (days)'] = swp['Battery Life (days)'].replace({'hours': '', 'days': '', 'Unlimited': 999}, regex=True)

# Converter a coluna 'Battery Life (days)' para numérico
swp['Battery Life (days)'] = pd.to_numeric(swp['Battery Life (days)'], errors='coerce')

# Definir o atributo preditor e o alvo
X = swp[['Battery Life (days)']].values  # Atributo relevante definido anteriormente
y = swp['Price (USD)'].values  # Alvo (Price USD)

# Remover valores NaN
mask = ~np.isnan(X).ravel() & ~np.isnan(y)
X = X[mask].reshape(-1, 1)
y = y[mask]

# Criar o modelo de regressão linear
linear_model = LinearRegression()

# Realizar a validação cruzada com K-Fold (5 divisões)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_linear = cross_val_score(linear_model, X, y, cv=kf, scoring='r2')

# Calcular a média e o desvio padrão dos scores R²
mean_cv_score_linear = np.mean(cv_scores_linear)
std_cv_score_linear = np.std(cv_scores_linear)

print("Regressão Linear:")
print("Score médio (R²) da regressão linear com K-Fold:", mean_cv_score_linear)
print("Desvio padrão dos scores R²:", std_cv_score_linear)

# Criar os modelos Lasso e Ridge
lasso_model = Lasso(alpha=0.0001)  # Valor de alpha obtido na questão 3
ridge_model = Ridge(alpha=0.0001)  # Valor de alpha obtido na questão 3

# Realizar a validação cruzada com K-Fold para Lasso
cv_scores_lasso = cross_val_score(lasso_model, X, y, cv=kf, scoring='r2')
mean_cv_score_lasso = np.mean(cv_scores_lasso)
std_cv_score_lasso = np.std(cv_scores_lasso)

# Realizar a validação cruzada com K-Fold para Ridge
cv_scores_ridge = cross_val_score(ridge_model, X, y, cv=kf, scoring='r2')
mean_cv_score_ridge = np.mean(cv_scores_ridge)
std_cv_score_ridge = np.std(cv_scores_ridge)

# Comparação dos resultados
print("\nLasso:")
print("Score médio (R²):", mean_cv_score_lasso)
print("Desvio padrão dos scores R²:", std_cv_score_lasso)

print("\nRidge:")
print("Score médio (R²):", mean_cv_score_ridge)
print("Desvio padrão dos scores R²:", std_cv_score_ridge)

# Comparação dos resultados dos três modelos
print("\nComparação dos Modelos:")
print("Regressão Linear - Média R²:", mean_cv_score_linear, "Desvio Padrão:", std_cv_score_linear)
print("Lasso - Média R²:", mean_cv_score_lasso, "Desvio Padrão:", std_cv_score_lasso)
print("Ridge - Média R²:", mean_cv_score_ridge, "Desvio Padrão:", std_cv_score_ridge)

# Escolha do melhor regressor baseado nos resultados:
if mean_cv_score_linear >= mean_cv_score_lasso and mean_cv_score_linear >= mean_cv_score_ridge:
    print("\nRegressão Linear é o modelo mais adequado.")
elif mean_cv_score_lasso > mean_cv_score_linear and mean_cv_score_lasso >= mean_cv_score_ridge:
    print("\nLasso é o modelo mais adequado.")
else:
    print("\nRidge é o modelo mais adequado.")



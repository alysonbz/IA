"""
- Questão 3

    Remova os atributos que não são relevantes para o processo de regressão e realize um gridsearch cross-validation para verificar
    qual a melhor parametrização para os regressores de Lasso e Ridge. Print as melhores configurações de cada um mostre também os melhores scores.

    Obs: Registrar na seção de resultados a análise realizada e discutir sobre os resultados encontrados.
"""

from src.utils import load_activision_blizzard_dataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

act_blz = load_activision_blizzard_dataset()

# Remover a coluna Date, pois não é relevante para o processo de regressão
act_blz_new = act_blz.drop(columns=['Date'])

# Definir a variável alvo (Adj Close) e as variáveis preditoras (demais colunas)
X = act_blz_new.drop(columns=['Adj Close'])
y = act_blz_new['Adj Close']

# Dividir os dados em conjunto de treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar as variáveis preditoras
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir os modelos Lasso e Ridge
lasso = Lasso(max_iter=10000)
ridge = Ridge()

# Definir o grid de parâmetros para ambos os modelos
param_grid = {
    'alpha': [0.01, 0.1, 1, 10, 100]
}

# Realizar GridSearchCV para Lasso
lasso_grid = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train_scaled, y_train)

# Realizar GridSearchCV para Ridge
ridge_grid = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train_scaled, y_train)

# Obter os melhores parâmetros e scores
lasso_best_params = lasso_grid.best_params_
ridge_best_params = ridge_grid.best_params_

lasso_best_score = -lasso_grid.best_score_
ridge_best_score = -ridge_grid.best_score_

print("Melhores parâmetros de Lasso:", lasso_best_params, "\nMelhor score de Lasso:", lasso_best_score)
print()

print("Melhores parâmetros de Ridge:", ridge_best_params, "\nMelhor score de Ridge:", ridge_best_score)

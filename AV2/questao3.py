import numpy as np
from src.utils import load_ferrari_dataset
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV


# Lendo o dataset.
ferrari = load_ferrari_dataset()


# Definir os atributos relevantes (mantendo apenas 'Close') e o alvo ('Adj Close').
X = ferrari[['Close']].values
y = ferrari['Adj Close'].values

# Definir os hiperparâmetros para Lasso e Ridge.
param_grid = {'alpha': np.logspace(-4, 4, 50)}

# Configurar o GridSearchCV para Lasso e Ridge.
lasso = Lasso()
ridge = Ridge()

# Busca os melhores valores do hiperparâmetro para os modelos Lasso e Ridge.
lasso_grid = GridSearchCV(lasso, param_grid, cv=5, scoring='r2')
ridge_grid = GridSearchCV(ridge, param_grid, cv=5, scoring='r2')

# Treinar os modelos utilizando GridSearchCV.
lasso_grid.fit(X, y)
ridge_grid.fit(X, y)

# Extrair os melhores parâmetros e scores.
best_lasso_params = lasso_grid.best_params_
best_lasso_score = lasso_grid.best_score_

# Armazena os melhores parâmetros encontrados pelo GridSearchCV.
best_ridge_params = ridge_grid.best_params_

# Armazena o melhor score (pontuação) obtido pelo modelo de Ridge.
best_ridge_score = ridge_grid.best_score_

# Extrair os melhores parâmetros e scores.
print("Melhores parâmetros Lasso:", best_lasso_params)
print("Melhor score Lasso:", best_lasso_score)
print("Melhores parâmetros Ridge:", best_ridge_params)
print("Melhor score Ridge:", best_ridge_score)
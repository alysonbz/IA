import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV


# Carregar o dataset
df = pd.read_csv("LG Electronics (2023-24).csv")


# Definir os atributos relevantes (mantendo apenas 'Close') e o alvo ('Adj Close')
X = df[['Close']].values
y = df['Adj Close'].values

# Definir os hiperparâmetros para Lasso e Ridge
param_grid = {'alpha': np.logspace(-4, 4, 50)}

# Configurar o GridSearchCV para Lasso e Ridge
lasso = Lasso()
ridge = Ridge()

lasso_grid = GridSearchCV(lasso, param_grid, cv=5, scoring='r2')
ridge_grid = GridSearchCV(ridge, param_grid, cv=5, scoring='r2')

# Treinar os modelos utilizando GridSearchCV
lasso_grid.fit(X, y)
ridge_grid.fit(X, y)

# Extrair os melhores parâmetros e scores
best_lasso_params = lasso_grid.best_params_
best_lasso_score = lasso_grid.best_score_

best_ridge_params = ridge_grid.best_params_
best_ridge_score = ridge_grid.best_score_

print("Melhores parâmetros Lasso:", best_lasso_params)
print("Melhor score Lasso:", best_lasso_score)
print("Melhores parâmetros Ridge:", best_ridge_params)
print("Melhor score Ridge:", best_ridge_score)

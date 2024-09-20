"""
### Questao3.py```

Remova os atributos que não são relevantes para o processo de regressão e realize um gridsearch cross-validation para verificar
qual a melhor parametrização para os regressores de Lasso e Ridge. Print as melhores configurações de cada um mostre também os melhores scores.
Obs: Registrar na seção de resultados a análise realizada e discutir sobre os resultados encontrados.

"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from src.utils import load_smart_watch_prices_dataset

# Carregar o dataset utilizando a função de utilidade
swp = load_smart_watch_prices_dataset()

# Remover o símbolo de dólar e converter 'Price (USD)' para numérico
swp['Price (USD)'] = swp['Price (USD)'].replace({r'\$': '', ',': ''}, regex=True).astype(float)

# Substituir valores como 'Unlimited' por um valor alto (999) e remover 'hours' e 'days'
swp['Battery Life (days)'] = swp['Battery Life (days)'].replace({'hours': '', 'days': '', 'Unlimited': 999}, regex=True)

# Converter a coluna 'Battery Life (days)' para numérico
swp['Battery Life (days)'] = pd.to_numeric(swp['Battery Life (days)'], errors='coerce')

# Remover colunas irrelevantes para o processo de regressão
# Vamos manter apenas atributos relevantes numéricos
X = swp[['Battery Life (days)']].values  # Aqui estamos considerando apenas "Battery Life (days)" como relevante
y = swp['Price (USD)'].values

# Remover valores NaN
mask = ~np.isnan(X).ravel() & ~np.isnan(y)
X = X[mask].reshape(-1, 1)
y = y[mask]

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

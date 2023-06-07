### Questão 3

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge

df = pd.read_csv(r'C:\Users\Guilherme\Documents\G\IA\AV2\Sample - Superstore.csv', encoding='latin-1')

#Remova os atributos que não são relevantes para o processo de regressão.
df_importante = df[['Sales'] + ['Profit']].copy()
print(df_importante)

#Realize um gridsearch cross-validation para verificar
#qual a melhor parametrização para os regressores de Lasso e Ridge.
X = df['Sales'].values.reshape(-1, 1)
y = df['Profit'].values

param_grid_lasso = {'alpha': np.arange(0.0001, 1, 10)}
param_grid_ridge = {"alpha": np.arange(0.0001, 1, 10)}

lasso = Lasso()
grid_search_lasso = GridSearchCV(lasso, param_grid_lasso, cv=5)
grid_search_lasso.fit(X, y)

ridge = Ridge()
grid_search_ridge = GridSearchCV(ridge, param_grid_ridge, cv=5)
grid_search_ridge.fit(X, y)

#Print as melhores configurações de cada um mostre também os melhores scores.
print("Melhores parâmetros para o regressor Lasso:\n", grid_search_lasso.best_params_)
print("Melhor score para o regressor Lasso:\n", grid_search_lasso.best_score_)
print("Melhores parâmetros para o regressor Ridge:\n", grid_search_ridge.best_params_)
print("Melhor score para o regressor Ridge:\n", grid_search_ridge.best_score_)

#Obs: Registrar na seção de resultados a análise realizada e discutir sobre os resultados encontrados.


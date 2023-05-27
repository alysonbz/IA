import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import Lasso, Ridge

# Carregar o dataset
data = pd.read_csv(r'C:\Users\eryka\Downloads\archive\Samsung Electronics.csv')

# Selecionar os atributos relevantes
relevant_attributes = ['High', 'Low', 'Close']  # Exemplo, substitua pelos atributos relevantes identificados

# Filtrar o DataFrame com os atributos relevantes
data_relevant = data[relevant_attributes]

# Separar os dados em atributos e alvo
X = data_relevant.drop(['Close', 'High'], axis=1).values
y = data_relevant[['Close','High']].values

# Definir os parâmetros para o grid search
param_grid = {'alpha': [0.1, 1, 10]}

# Realizar o grid search para o regressor Lasso
lasso_model = Lasso()
lasso_grid_search = GridSearchCV(lasso_model, param_grid, cv=5)
lasso_grid_search.fit(X, y)

# Imprimir as melhores configurações e o melhor score para o regressor Lasso
print("Melhores configurações para Lasso:", lasso_grid_search.best_params_)
print("Melhor score para Lasso:", lasso_grid_search.best_score_)

# Realizar o grid search para o regressor Ridge
ridge_model = Ridge()
ridge_grid_search = GridSearchCV(ridge_model, param_grid, cv=5)
ridge_grid_search.fit(X, y)

# Imprimir as melhores configurações e o melhor score para o regressor Ridge
print("Melhores configurações para Ridge:", ridge_grid_search.best_params_)
print("Melhor score para Ridge:", ridge_grid_search.best_score_)

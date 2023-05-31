import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import Lasso, Ridge, LinearRegression

# Carregar o dataset atualizado
data = pd.read_csv("forest_fires_updated.csv")

# Selecionar os atributos relevantes
relevant_columns = ['month', 'day', 'ISI', 'temp', 'wind']
X = data[relevant_columns].values
y = data['area'].values

# Definir o número de folds para k-fold
n_folds = 5

# Criar o objeto de k-fold
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Realizar a regressão linear com Lasso usando cross-validation
lasso = Lasso(alpha=10.0) #alpha=10 que foi o melhor da 3
lasso_scores = cross_val_score(lasso, X, y, cv=kf)
print("Scores de cross-validation para Lasso:", lasso_scores)
print("Média dos scores de cross-validation para Lasso:", np.mean(lasso_scores))

# Realizar a regressão linear com Ridge usando cross-validation
ridge = Ridge(alpha=10.0) #alpha=10 que foi o melhor da 3
ridge_scores = cross_val_score(ridge, X, y, cv=kf)
print("Scores de cross-validation para Ridge:", ridge_scores)
print("Média dos scores de cross-validation para Ridge:", np.mean(ridge_scores))

#simples usando cross validation
linear_regression = LinearRegression()
linear_regression_scores = cross_val_score(linear_regression, X, y, cv=kf)
print("Scores de cross-validation para Regressão Linear:", linear_regression_scores)
print("Média dos scores de cross-validation para Regressão Linear:", np.mean(linear_regression_scores))
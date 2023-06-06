import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import LabelEncoder

# Carregar o dataset
dados = pd.read_csv('car_price_prediction.csv')

# Selecionar os atributos relevantes
relevant_columns = ['Engine volume']
X = dados[relevant_columns]
y = dados['Price']

# Realizar a codificação de atributos categóricos, se necessário
label_encoder = LabelEncoder()
X['Engine volume'] = label_encoder.fit_transform(X['Engine volume'])

# Definir o número de folds para k-fold
n_folds = 5

# Criar o objeto de k-fold
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Realizar a regressão linear com Lasso usando cross-validation
lasso = Lasso(alpha=0.1)  # Utilize o valor de alpha obtido na questão 4 para Lasso
lasso_scores = cross_val_score(lasso, X, y, cv=kf)
print("Scores de cross-validation para Lasso:", lasso_scores)
print("Média dos scores de cross-validation para Lasso:", np.mean(lasso_scores))

# Realizar a regressão linear com Ridge usando cross-validation
ridge = Ridge(alpha=0.1)  # Utilize o valor de alpha obtido na questão 4 para Ridge
ridge_scores = cross_val_score(ridge, X, y, cv=kf)
print("Scores de cross-validation para Ridge:", ridge_scores)
print("Média dos scores de cross-validation para Ridge:", np.mean(ridge_scores))
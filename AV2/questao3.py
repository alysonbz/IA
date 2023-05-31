'''Remova os atributos que não são relevantes para o processo de regressão e realize um gridsearch
 cross-validation para verificar qual a melhor parametrização para os regressores de Lasso e Ridge.
 Print as melhores configurações de cada um mostre também os melhores scores. Obs: Registrar na seção
 de resultados a análise realizada e discutir sobre os resultados encontrados.
'''

# Importando as bibliotecas necessárias
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import numpy as np
from questao2 import car_price

car_price = car_price
print(car_price)
X = car_price.drop("price", axis=1).values # todas encoder, exceto o atributo alvo "price"
y = car_price["price"].values # preço
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Inicializando o Lasso
lasso = Lasso()

# Inicializando o KFold
kf = KFold(n_splits=6, shuffle=True, random_state=42)

# Configurando os parâmetros
param_grid = {"alpha": np.arange(0.00001, 1, 20)}

# Instanciando lasso_cv
lasso_cv = GridSearchCV(lasso, param_grid, cv = kf)

# Ajustar aos dados de treinamento
lasso_cv.fit(X_train, y_train)

print("Parâmetros de Lasso ajustados: {}".format(lasso_cv.best_params_))
print("Pontuação de Lasso afinado: {}".format(lasso_cv.best_score_))


# Inicializando o Ridge

param_grid_2 = {"alpha": np.arange(0.00001, 1, 20),
              "solver":["sag","lsqr"]}

# Instanciando Ridge
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid_2, cv=kf)
ridge_cv.fit(X_train,y_train)

print("Parâmetros de Ridge ajustados: {}".format(ridge_cv.best_params_))
print("Pontuação de Ridge afinado: {}".format(ridge_cv.best_score_))
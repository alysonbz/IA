#bibliotecas
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
dados = pd.read_csv('dados.csv')
df = pd.read_csv('dados1.csv')



X = df.drop(['Price'], axis=1)
y = df['Price'].values


# Realizar a codificação de atributos categóricos, se necessário
label_encoder = LabelEncoder()
X['Engine volume'] = label_encoder.fit_transform(X['Engine volume'])

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir os parâmetros para busca em grade
param_grid = {'alpha': [0.1, 1.0, 10.0]}

# Realizar busca em grade com validação cruzada para o regressor Lasso
lasso = Lasso()
lasso_cv = GridSearchCV(lasso, param_grid, cv=5)
lasso_cv.fit(X_train, y_train)

# Imprimir os melhores parâmetros e score para o regressor Lasso
print("Melhores parâmetros para Lasso:", lasso_cv.best_params_)
print("Melhor score para Lasso:", lasso_cv.best_score_)

# Realizar busca em grade com validação cruzada para o regressor Ridge
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv=5)
ridge_cv.fit(X_train, y_train)

# Imprimir os melhores parâmetros e score para o regressor Ridge
print("Melhores parâmetros para Ridge:", ridge_cv.best_params_)
print("Melhor score para Ridge:", ridge_cv.best_score_)



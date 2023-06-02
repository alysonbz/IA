from questao1 import database
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

#Remova os atributos que não são relevantes para o processo de  regressão e realize um gridsearch
#cross-validation para verificar qual a melhor parametrização para os regressores de Lasso e Ridge.
relevant_columns = ['duration', 'price']
data_relevant = database[relevant_columns]

X = data_relevant.drop('price', axis=1)
y = data_relevant['price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

parameters = {'alpha': [0.01, 0.1, 1, 10]}

lasso = Lasso()
ridge = Ridge()

lasso_grid = GridSearchCV(lasso, parameters, cv=5)
lasso_grid.fit(X_train, y_train)

ridge_grid = GridSearchCV(ridge, parameters, cv=5)
ridge_grid.fit(X_train, y_train)

#Print as melhores configurações de cada um mostre também os melhores scores. Obs: Registrar na
#seção de resultados a análise realizada e discutir sobre os resultados encontrados.

print("Melhores configurações para Lasso:")
print(lasso_grid.best_params_)
print("Melhor score para Lasso:")
print(lasso_grid.best_score_)

print("Melhores configurações para Ridge:")
print(ridge_grid.best_params_)
print("Melhor score para Ridge:")
print(ridge_grid.best_score_)




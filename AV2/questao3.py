import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Carregar o dataset
clean_dataset = pd.read_csv('./dataset/Clean_Dataset.csv')


clean_dataset["airline"] = clean_dataset["airline"].replace("SpiceJet", 0)
clean_dataset["airline"] = clean_dataset["airline"].replace("AirAsia", 1)
clean_dataset["airline"] = clean_dataset["airline"].replace("Vistara", 2)
clean_dataset["airline"] = clean_dataset["airline"].replace("GO_FIRST", 3)
clean_dataset["airline"] = clean_dataset["airline"].replace("Indigo", 4)
clean_dataset["airline"] = clean_dataset["airline"].replace("Air_India", 5)

clean_dataset["departure_time"] = clean_dataset["departure_time"].replace("Evening", 0)
clean_dataset["departure_time"] = clean_dataset["departure_time"].replace("Early_Morning", 1)
clean_dataset["departure_time"] = clean_dataset["departure_time"].replace("Morning", 2)
clean_dataset["departure_time"] = clean_dataset["departure_time"].replace("Afternoon", 3)
clean_dataset["departure_time"] = clean_dataset["departure_time"].replace("Night", 4)
clean_dataset["departure_time"] = clean_dataset["departure_time"].replace("Late_Night", 5)


clean_dataset["stops"] = clean_dataset["stops"].replace("zero", 0)
clean_dataset["stops"] = clean_dataset["stops"].replace("one", 1)
clean_dataset["stops"] = clean_dataset["stops"].replace("two_or_more", 2)

clean_dataset["class"] = clean_dataset["class"].replace("Economy", 0)
clean_dataset["class"] = clean_dataset["class"].replace("Business", 1)


categorical_columns = ['flight', 'source_city',
                       'arrival_time', 'destination_city']
numerical_columns = ['duration','stops','departure_time','airline', 'days_left', 'class']

# Definir a variável alvo (Price) e as variáveis preditoras (demais colunas)
X = clean_dataset.drop(columns=['price'])  # A variável 'price' é a variável alvo
y = clean_dataset['price']  # Alvo

# Dividir os dados em conjunto de treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir o pré-processamento para as variáveis categóricas e numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),  # Escalar as colunas numéricas
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)  # One-Hot Encoding para as colunas categóricas
    ])

# Definir os modelos Lasso e Ridge
lasso = Lasso(max_iter=10000)
ridge = Ridge()

# Criar pipelines para Lasso e Ridge, aplicando o pré-processamento
lasso_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', lasso)])
ridge_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', ridge)])

# Definir o grid de parâmetros para ambos os modelos
param_grid = {
    'model__alpha': [0.01, 0.1, 1, 10, 100]
}

# Realizar GridSearchCV para Lasso
lasso_grid = GridSearchCV(lasso_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)

# Realizar GridSearchCV para Ridge
ridge_grid = GridSearchCV(ridge_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train, y_train)

# Obter os melhores parâmetros e scores
lasso_best_params = lasso_grid.best_params_
ridge_best_params = ridge_grid.best_params_

lasso_best_score = -lasso_grid.best_score_
ridge_best_score = -ridge_grid.best_score_

# Imprimir os resultados
print("Melhores parâmetros de Lasso:", lasso_best_params)
print("Melhor score de Lasso (MSE):", lasso_best_score)
print()

print("Melhores parâmetros de Ridge:", ridge_best_params)
print("Melhor score de Ridge (MSE):", ridge_best_score)

# Avaliação com os dados de teste
y_pred_lasso = lasso_grid.best_estimator_.predict(X_test)
y_pred_ridge = ridge_grid.best_estimator_.predict(X_test)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

print(f"MSE de Lasso no conjunto de teste: {mse_lasso}")
print(f"MSE de Ridge no conjunto de teste: {mse_ridge}")


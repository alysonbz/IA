import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

# Carregar o dataset
Lenovo = pd.read_csv('lenovo.csv')

# Separar os atributos de entrada (X) e o atributo alvo (y)
X = Lenovo.drop(['Open'], axis=1)
y = Lenovo['Open']

# Inicializar o regressor LinearRegression
regressor = LinearRegression()

# Inicializar o objeto KFold para validação cruzada
kf = KFold(n_splits=6, shuffle=True, random_state=42)

# Realizar a validação cruzada com o regressor LinearRegression
scores = cross_val_score(regressor, X, y, cv=kf, scoring='r2')

# Imprimir os scores de cada fold
print("Scores de cada fold:", scores)

# Calcular a média dos scores
mean_score = np.mean(scores)
print("Média dos scores:", mean_score)

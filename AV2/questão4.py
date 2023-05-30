#bibliotecas
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold


#Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
dm = pd.read_csv('dados.csv')
print(dm)

# Separar os atributos de entrada (X) e o atributo alvo (y)
X = dm.drop(['Price'], axis=1)
y = dm['Price']



# Inicializar o modelo de regressão linear
regressor = LinearRegression()

# Inicializar o objeto KFold para validação cruzada
kf = KFold(n_splits=6, shuffle=True, random_state=42)

# Realizar a validação cruzada e obter os scores
cv_scores = cross_val_score(regressor, X, y, cv=kf)

# Imprimir os scores para cada fold
print("Scores para cada fold:")
print(cv_scores)

# Calcular a média dos scores
mean_score = cv_scores.mean()
print("Média dos scores:", mean_score)

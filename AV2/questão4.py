import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Carregar o dataset
data = pd.read_csv(r'C:\Users\eryka\Downloads\archive\Samsung Electronics.csv')

# Separar atributo alvo e atributo relevante
X = data[['Close']]
y = data[['Close']]

# Criar objeto do modelo de regressão linear
model = LinearRegression()

# Realizar a validação cruzada utilizando k-fold
scores = cross_val_score(model, X, y, cv=5)

# Imprimir os scores de cada fold
for i, score in enumerate(scores):
    print("Fold", i+1, ":", score)

# Imprimir o score médio
print("Score médio:", np.mean(scores))

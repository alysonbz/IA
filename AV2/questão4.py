### Questão 4

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

df = pd.read_csv(r'C:\Users\Guilherme\Documents\G\IA\AV2\Sample - Superstore.csv', encoding='latin-1')

#Utilizando kfold e cross-validation faça uma regressão linear utilizando os mesmos atributos definidos na questão 3.
X = df['Sales'].values.reshape(-1, 1)
y = df['Profit'].values
kf = KFold(n_splits=6, shuffle=True, random_state=5)
reg = LinearRegression()
cv_scores = cross_val_score(reg, X, y, cv=kf)
print("Scores:\n", cv_scores)
print("Média: ", np.mean(cv_scores))

#Obs: Com os resultados obtidos na questão 3 e da questão 4 faça uma comparação entre os desempenhos.
#Escolha o regressor adequado e informe o motivo da escolha. Discuta sobre as limitações e acertos encontrados.
 
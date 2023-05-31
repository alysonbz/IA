'''Utilizando kfold e cross-validation faça uma regressão linear utilizando os mesmos atributos
definidos na questão 3. Obs: Com os resultados obtidos na questão 3 e da questão 4 faça uma
comparação entre os desempenhos. Escolha o regressor adequado e informe o motivo da escolha.
Discuta sobre as limitações e acertos encontrados.'''

# Importando as bibliotecas necessárias
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from questao2 import car_price

car_price = car_price
print(car_price)

# Criar X e y
X = car_price.drop("price", axis=1).values # todas encoder, exceto o atributo alvo "price"
y = car_price["price"].values # preço


# Criar um objeto do KFold
kf = KFold(n_splits=6, shuffle=True, random_state=5)

reg = LinearRegression()

# Calcular pontuações de validação cruzada 6 vezes
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Print cv_scores
print("Score validação cruzada: ", cv_scores)

# Print da média
print("Média: ", np.mean(cv_scores)) # média das pontuações

# Print do desvio padrão
print("Desvio padrão: ", np.std(cv_scores))

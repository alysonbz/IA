import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# Carregar o dataset
dados = pd.read_csv('Mobile phone price.csv')

# Verificar as colunas disponíveis no DataFrame
print(dados.columns)

# Verificar quais colunas são as mais relevantes e criar um novo DataFrame
colunas_relevantes = ['Brand', 'Model', 'Storage ', 'RAM ','Price']

dm = dados[colunas_relevantes].copy()

# Criar uma instância do LabelEncoder
label_encoder = LabelEncoder()

# Aplicar o LabelEncoder às colunas categóricas
for coluna in colunas_relevantes:
    if dm[coluna].dtype == 'object':
        dm.loc[:, coluna] = label_encoder.fit_transform(dm[coluna])

# Exibir o DataFrame com as colunas codificadas
print(dm)

# Atributo mais relevante
X = dm.drop(['Price'], axis=1)
y = dm['Price'].values
dm_columns = X.columns

# Instanciar o modelo de regressão Lasso
lasso = Lasso(alpha=0.3)

# Calcular e imprimir os coeficientes
lasso_coef = lasso.fit(X, y).coef_
plt.bar(dm_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()

dm.to_csv("dados.csv", index=False)

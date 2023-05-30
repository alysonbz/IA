#importe as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder


## Carregue o dataset definido para você
dados = pd.read_csv('laptopPrice.csv')


# Verifique quais colunas são as mais relevantes e crie um novo dataframe
colunas_relevantes = ['brand', 'processor_brand', 'processor_name', 'processor_gnrtn', 'ram_gb', 'ram_type', 'ssd', 'Price', 'rating', 'Number of Ratings', 'Number of Reviews']
dm = dados[colunas_relevantes].copy()


# Criar uma instância do LabelEncoder
label_encoder = LabelEncoder()


# Aplicar o LabelEncoder às colunas categóricas
for coluna in colunas_relevantes:
    if dm[coluna].dtype == 'object':
        dm.loc[:, coluna] = label_encoder.fit_transform(dm[coluna])

# Exibir o DataFrame com as colunas codificadas
print(dm)


#atributo mais relevante
X = dm.drop(['Price'], axis=1)
y = dm['Price'].values
dm_columns = X.columns



# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)

# Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
plt.bar(dm_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()


dm.to_csv("dados.csv", index=False)

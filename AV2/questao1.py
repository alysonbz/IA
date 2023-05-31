#importe as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder


## Carregue o dataset definido para você
dados = pd.read_csv('CO2 Emissions_Canada.csv')
print(dados.info())


# Verifique quais colunas são as mais relevantes e crie um novo dataframe
colunas_relevantes = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']
dm = dados[colunas_relevantes].copy()


# Criar uma instância do LabelEncoder
label_encoder = LabelEncoder()

# Aplicar o LabelEncoder às colunas categóricas
for coluna in colunas_relevantes:
    if dm[coluna].dtype == 'object':
        dm.loc[:, coluna] = label_encoder.fit_transform(dm[coluna])


#atributo mais relevante
X = dm.drop(['CO2 Emissions(g/km)'], axis=1)
y = dm['CO2 Emissions(g/km)'].values
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
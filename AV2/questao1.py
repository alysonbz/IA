#bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder

##importação do dataset
dados = pd.read_csv('car_price_prediction.csv')

print(dados.columns)


#Verifique qual atributo será o alvo para regressão no seu dataset e faça uma análise de qual atributo é mais relevante para realizar
# a regressão do alvo escolhido.
#Lembre de comprovar via gráfico. Obs: Registrar na seção de resultados a análise realizada e discutir sobre o resultado encontrado.

#Vendo se tem Na ou Null

'''print(dados.isnull().sum())
print(dados.isna().sum())'''


# Verifique quais colunas são as mais relevantes e crie um novo dataframe
colunas_relevantes = ['Price', 'Manufacturer', 'Category', 'Leather interior', 'Fuel type', 'Engine volume', 'Mileage',
       'Cylinders', 'Gear box type', 'Drive wheels']
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


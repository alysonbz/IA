import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

data = pd.read_csv("forest_fires.csv")

# Criar arrays X e y
X = data.drop(["area"], axis=1)
y = data["area"].values
data_columns = X.columns

# Verificar tipos de dados antes da conversão
print("Tipos de dados antes da conversão:")
print(data.dtypes)

# Definindo os dicionários de mapeamento
day_mapping = {'sun': 1, 'mon': 2, 'tue': 3, 'wed': 4, 'thu': 5, 'fri': 6, 'sat': 7}
month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

# Aplicando o mapeamento nas colunas "day" e "month"
data['day'] = data['day'].map(day_mapping).astype(float)
data['month'] = data['month'].map(month_mapping).astype(float)

# Atualizar o DataFrame X após a conversão
X = data.drop(["area"], axis=1)

# Verificar tipos de dados após a conversão
print("Tipos de dados após a conversão:")
print(data.dtypes)

# Exibindo o DataFrame atualizado
print(data)

# Instantiate a lasso regression model
lasso = Lasso(alpha=0.1)

# Calcular e imprimir os coeficientes
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
plt.bar(data_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()



## Carregue o dataset definido para você



# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.



# Verifique quais colunas são as mais relevantes e crie um novo dataframe.



#Print o dataframe final e mostre a distribuição de classes que você deve classificar



#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário



#Salve o dataset atualizado se houver modificações.
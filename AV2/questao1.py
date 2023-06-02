#importe as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

## Carregue o dataset definido para você
ds=pd.read_csv("C:/Users/Luciana/OneDrive/Área de Trabalho/Clean_Dataset.csv")
print("Data Frame inicial:\n",ds)
print("Características do Data Frame:\n",ds.info())

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(ds.isna().sum(), '\n')

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
#==============================================================================================================================================================================
print("Frequência de cada valor presente na coluna:\n", ds['flight'].value_counts().to_string(),'\n','\n')
ds=ds.drop('flight', axis=1)
#==============================================================================================================================================================================
print("Frequência de cada valor presente na coluna:\n", ds['days_left'].value_counts(),'\n','\n')
ds=ds.drop('days_left', axis=1)
#==============================================================================================================================================================================

#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
#==============================================================================================================================================================================
print("Frequência de cada valor presente na coluna:\n", ds['airline'].value_counts(),'\n','\n')
ds['airline'] = ds['airline'].replace({'Vistara': 0, 'Air_India': 1, 'Indigo':2, 'GO_FIRST': 3, 'AirAsia': 4, 'SpiceJet':5})
#==============================================================================================================================================================================
print("Frequência de cada valor presente na coluna:\n", ds['source_city'].value_counts(),'\n','\n')
ds['source_city'] = ds['source_city'].replace({'Delhi': 0, 'Mumbai': 1, 'Bangalore':2, 'Kolkata': 3, 'Hyderabad': 4, 'Chennai':5})
#==============================================================================================================================================================================
print("Frequência de cada valor presente na coluna:\n", ds['departure_time'].value_counts(),'\n','\n')
ds['departure_time'] = ds['departure_time'].replace({'Morning': 0, 'Early_Morning': 1, 'Evening':2, 'Night': 3, 'Afternoon': 4, 'Late_Night':5})
#==============================================================================================================================================================================
print("Frequência de cada valor presente na coluna:\n", ds['stops'].value_counts(),'\n','\n')
ds['stops'] = ds['stops'].replace({'zero': 0, 'one': 1, 'two_or_more':2})
#==============================================================================================================================================================================
print("Frequência de cada valor presente na coluna:\n", ds['arrival_time'].value_counts(),'\n','\n')
ds['arrival_time'] = ds['arrival_time'].replace({'Night': 0, 'Evening': 1, 'Morning':2, 'Afternoon': 3, 'Early_Morning': 4, 'Late_Night':5})
#==============================================================================================================================================================================
print("Frequência de cada valor presente na coluna:\n", ds['destination_city'].value_counts(),'\n','\n')
ds['destination_city'] = ds['destination_city'].replace({'Mumbai': 0, 'Delhi': 1, 'Bangalore':2, 'Kolkata': 3, 'Hyderabad': 4, 'Chennai':5})
#==============================================================================================================================================================================
print("Frequência de cada valor presente na coluna:\n", ds['class'].value_counts(),'\n','\n')
ds['class'] = ds['class'].replace({'Economy': 0, 'Business': 1})
#==============================================================================================================================================================================
print("Frequência de cada valor presente na coluna:\n", ds['price'].value_counts().to_string(),'\n','\n')
#==============================================================================================================================================================================
print("Frequência de cada valor presente na coluna:\n", ds['duration'].value_counts().to_string(),'\n','\n')
ds['duration']=ds['duration'].astype(int)
print("Frequência de cada valor presente na coluna:\n", ds['duration'])
#==============================================================================================================================================================================

#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print("Data Frame após ajustes:\n",ds)
print("Características do Data Frame atualizado:\n",ds.info())

#Salve o dataset atualizado se houver modificações.
database=ds
print("Data Frame atualizado:\n",database)
database.to_csv=('C:/Users/Luciana/OneDrive/Área de Trabalho/Clean_Dataset.csv')

#Atributo mais importnate da dataset
X =  database.drop("price", axis= 1).values
y = database["price"].values
nomes = database.drop("price", axis= 1).columns
lasso = Lasso(alpha= 0.1)
lasso_coef = lasso.fit(X, y).coef_

fig = plt.figure(facecolor='#aebd93')

plt.bar(nomes, lasso_coef, color = "#440958", edgecolor = "#095810")
plt.xticks(rotation = 45)
plt.show()
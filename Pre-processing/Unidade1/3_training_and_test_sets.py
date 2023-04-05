from src.utils import load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()
print('\nExclui as colunas Latitude e Longitude de volunteer')
# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(['Latitude', 'Longitude'], axis = 1)
print(volunteer_new.info)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
print('\n')
volunteer = ___

# mostre o balanceamento das classes em 'category_desc'
print(___['category_desc'].__,'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.__(__, axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = __[['__']]

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = __(__, __, stratify=__, random_state=42)

# mostre o balanceamento das classes em 'category_desc' novamente
___
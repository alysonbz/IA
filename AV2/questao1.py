#importe as bibliotecas necessárias
import pandas as pd

## Carregue o dataset definido para você
dados = pd.read_csv('laptopPrice.csv')


# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
'''print(dados.isnull().sum())
print(dados.isna().sum())'''



# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
dm = dados[['brand', 'processor_brand', 'processor_name', 'processor_gnrtn', 'ram_gb', 'ram_type', 'ssd', 'Price', 'rating', 'Number of Ratings', 'Number of Reviews']]


#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print(dm['brand'].value_counts(),'\n','\n')
print(dm['processor_brand'].value_counts(),'\n','\n')
print(dm['processor_name'].value_counts(),'\n','\n')
print(dm['processor_gnrtn'].value_counts(),'\n','\n')
print(dm['ram_gb'].value_counts(),'\n','\n')
print(dm['ram_type'].value_counts(),'\n','\n')
print(dm['ssd'].value_counts(),'\n','\n')
print(dm['Price'].value_counts(),'\n','\n')
print(dm['rating'].value_counts(),'\n','\n')
print(dm['Number of Ratings'].value_counts(),'\n','\n')
print(dm['Number of Reviews'].value_counts(),'\n','\n')

#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário



#Salve o dataset atualizado se houver modificações.
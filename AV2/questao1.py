#importe as bibliotecas necessárias
import pandas as pd

## Carregue o dataset definido para você
df = pd.read_csv(r'C:\Users\LAB1_00\Desktop\Dataframe\HP_share_prices.csv')
print(df)

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(df.isnull().sum())


# Verifique quais colunas são as mais relevantes e crie um novo dataframe.



#Print o dataframe final e mostre a distribuição de classes que você deve classificar



#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário



#Salve o dataset atualizado se houver modificações.
#Allan Michel
#importe as bibliotecas necessárias
from src.utils import load_df1_unidade1
import pandas as pd
## Carregue o dataset definido para você
df1 = load_df1_unidade1()


# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(df1)
print(df1.dropna())


# Verifique quais colunas são as mais relevantes e crie um novo dataframe.



#Print o dataframe final e mostre a distribuição de classes que você deve classificar



#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário



#Salve o dataset atualizado se houver modificações.
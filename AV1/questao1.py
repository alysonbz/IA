#importe as bibliotecas necessárias
import pandas as pd

## Carregue o dataset definido para você
df = pd.read_csv('customer_data.csv')

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
if df.isna().any().any():
    df = df.dropna()

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.


#Print o dataframe final e mostre a distribuição de classes que você deve classificar
counts = df['label'].value_counts()
print('\nDistribuição de classes:\n', counts)

#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
#Não é necessário renomear a coluna "label" se ela já representa a classe do problema.

#Salve o dataset atualizado se houver modificações.
customer_ajustado = df
customer_ajustado.to_csv = ('customer_ajustado.csv')
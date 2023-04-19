#importe as bibliotecas necessárias
import pandas as pd

## Carregue o dataset definido para você
df = pd.read_csv('customer_data.csv')

print(df)

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
# verifica se há valores faltantes (NaN ou células vazias) no DataFrame
if df.isna().any().any():
    # exclui as linhas com valores faltantes
    df = df.dropna()

# preenche os valores faltantes restantes com zero

print(df)

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.



#Print o dataframe final e mostre a distribuição de classes que você deve classificar
counts = df['label'].value_counts()
print('\nDistribuição de classes:\n', counts)


#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
#Não é necessário renomear a coluna "label" se ela já representa a classe do problema.


#Salve o dataset atualizado se houver modificações.

customer_ajustado = df
customer_ajustado.to_csv = ('customer_ajustado.csv')
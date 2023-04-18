#importe as bibliotecas necessárias


import pandas as pd

## Carregue o dataset definido para você

dados_cancer = pd.read_csv('breast-cancer.csv')
#print(dados_cancer)
# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(dados_cancer.isna().sum())


# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
X = dados_cancer.drop('diagnosis', axis=1)


#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print(X.info())
print(X[''].value_counts(),'\n','\n')

#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário



#Salve o dataset atualizado se houver modificações.
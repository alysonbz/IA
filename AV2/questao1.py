#importe as bibliotecas necessárias
import pandas as pd

## Carregue o dataset definido para você
smarthwatch = pd.read_csv('Smartwatchprices.csv')
print("\n Dataset: Smart watch prices")
print(smarthwatch)


# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print("\n Verificação da existência de células vaizas ou Nan")
print(smarthwatch.isna().sum())

print("\n Dropando Células Vazias")
smarthwatch_sem_na = smarthwatch.dropna()
print(smarthwatch_sem_na.isna().sum())

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.



#Print o dataframe final e mostre a distribuição de classes que você deve classificar



#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário



#Salve o dataset atualizado se houver modificações.
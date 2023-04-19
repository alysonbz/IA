#importe as bibliotecas necessárias
import pandas as pd

## Carregue o dataset definido para você
read_cancer = pd.read_csv(r"C:\Users\LAB1_00\Documents\Livia\IA\AV1\dataset\Cancer_Data.csv")


# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(read_cancer.isnull().sum())


# Verifique quais colunas são as mais relevantes e crie um novo dataframe.



#Print o dataframe final e mostre a distribuição de classes que você deve classificar



#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário



#Salve o dataset atualizado se houver modificações.
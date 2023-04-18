#importe as bibliotecas necessárias

import pandas as pd
import numpy as np


## Carregue o dataset definido para você
diabetes = pd.read_csv("..\AV1\dataset\diabetes_012_health_indicators_BRFSS2015.csv")
print(diabetes)

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(diabetes.head())
print(diabetes.isna().sum())
print(diabetes.isnull().sum())
print(diabetes.info)
# Verifique quais colunas são as mais relevantes e crie um novo dataframe.



#Print o dataframe final e mostre a distribuição de classes que você deve classificar



#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário



#Salve o dataset atualizado se houver modificações.
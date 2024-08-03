#importe as bibliotecas necessárias

import pandas as pd
import numpy as np
from src.utils import load_gender_classification_dataset

## Carregue o dataset definido para você
db = load_gender_classification_dataset()
db.head()


# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
db_limpo = db.dropna()
db_limpo.head()



# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
db_relevante = db_limpo[['long_hair', 'forehead_width_cm', 'forehead_height_cm', 'nose_wide', 'nose_long', 'lips_thin', 'distance_nose_to_lip_long', 'gender']]



#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print(db_relevante.head())
print(db_relevante['gender'].value_counts())



#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
db_relevante['gender'] = db_relevante['gender'].map({'Male': 1, 'Female': 0})

#Salve o dataset atualizado se houver modificações.
db_relevante.to_csv('dataset/gender_classification_v7.csv', index=False)


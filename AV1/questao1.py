#importe as bibliotecas necessárias
import pandas as pd
import numpy as np


## Carregue o dataset definido para você
dados_avc = pd.read_csv('healthcare-dataset-stroke-data.csv')
print(dados_avc)

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.

       # transformando a coluna 'smoking_status' em numérico:
def transf_smoking_status():
    dados_avc = pd.read_csv('healthcare-dataset-stroke-data.csv')
    le = LabelEnconder()
    dados_avc["smoking_status"] = le.fit_transform(dados_avc["smoking_status"])
    return dados_avc

transf_smoking_status()

      # transformando a coluna 'smoking_status' em numérico:
def transf_gender():
    dados_avc = pd.read_csv('healthcare-dataset-stroke-data.csv')
    le = LabelEnconder()
    dados_avc["gender"] = le.fit_transform(dados_avc["gender"])
    return dados_avc
transf_gender()

print(dados_avc.isna().sum())
dados_avc_ajustados = dados_avc.dropna(subset=["bmi"])
print(dados_avc_ajustados.isna().sum())

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
dados_avc_relevantes = dados_avc_ajustados.drop("id", "ever_married", "work_type", "Residence_type", axis=1)
print(dados_avc_relevantes)

# Print o dataframe final e mostre a distribuição de classes que você deve classificar



# Observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário



# Salve o dataset atualizado se houver modificações.
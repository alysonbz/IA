#importe as bibliotecas necessárias
import pandas as pd

## Carregue o dataset definido para você
gender = pd.read_csv('gender_classification_v7.csv')
print("\n Dataset: Gender Classification")
print(gender)


# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print("\n Verificação da existência de células vaizas ou Nan")
print(gender.isna().sum())
print("Não há NAN nem celulas vazias neste dataset")


# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
print("\n Restrição de colunas mais relevante no dataset")
gender_relevantes = gender[["long_hair", "forehead_width_cm", "forehead_height_cm", "nose_wide", "nose_long", "lips_thin", "gender"]]


#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print(gender_relevantes)
print("\nUtilizarei as colunas 'gender', após modifica-la")

#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
print("\nTransformando a coluna 'gender' em int:")
gender_atualiza = pd.get_dummies(gender_relevantes["gender"])
print(gender_atualiza)
gender_atualizando = pd.concat((gender_atualiza, gender_relevantes), axis=1)
print("\nApagando as colunas desatualizadas:")
gender_atualizando = gender_atualizando.drop(["gender"], axis=1)
gender_atualizando = gender_atualizando.drop(["Male"], axis=1)
gender_atualizado = gender_atualizando.rename(columns={"Female": "gender"})
print("Dataset atualizado:")
print(gender_atualizado)

#Voltando a Distribuição após renomear os atributos
print("Distribuição:\n", gender_atualizado['gender'].value_counts())

#Salve o dataset atualizado se houver modificações.
gender_atualizado.to_csv("gender_final.csv")
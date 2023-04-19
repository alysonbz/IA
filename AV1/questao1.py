#importe as bibliotecas necessárias

import pandas as pd


## Carregue o dataset definido para você
diabetes = pd.read_csv(r"C:\Users\LAB1_00\Desktop\BIANCA\ia_novo\IA\AV1\dataset\diabetes_012_health_indicators_BRFSS2015.csv")
print(diabetes.shape)

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(diabetes.isna().sum())
print(diabetes.isnull().sum())

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
#excluindo as colunas que eu não quero
diabetes = diabetes.drop(["CholCheck", "Education", "Income", "MentHlth", "GenHlth", "NoDocbcCost", "Smoker",
                          "AnyHealthcare", "DiffWalk","Fruits", "Veggies", "Stroke", "HeartDiseaseorAttack"], axis=1)

#traduzindo o nome das colunas
diabetes_ajustado = diabetes.rename(columns={"Diabetes_012":"Diabetes",
                                    "HighBP": "Pressão_Alta",
                                    "HighChol": "Colesterol_alto",
                                    "BMI": "IMC",
                                    "Smoker": "Fumantes",
                                    "PhysActivity": "Atividade_fisica",
                                    "HvyAlcoholConsump": "Consumo_Alcool",
                                    "PhysHlth": "Saude_fisica",
                                    "Sex": "Genero",
                                    "Age": "Idade"})

print(diabetes_ajustado.shape)

#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print(diabetes_ajustado["Diabetes"].value_counts())


#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
# O dataset já estava categorizado. 0 para quem não tem Diabetes, 1 para diabetes na gravidez e 2 para diabetes normal


#Salve o dataset atualizado se houver modificações.
diabetes_ajustado.to_csv((r"C:\Users\LAB1_00\Desktop\BIANCA\ia_novo\IA\AV1\dataset\diabetes_ajustado.csv"))

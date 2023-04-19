#importe as bibliotecas necessárias
import pandas as pd

## Carregue o dataset definido para você
db = pd.read_csv("diabetes.csv")
print("\n Base de dados:")
print(db)


# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print("\n Verificando se existem células vazias ou Nan:")
print(db.isna().sum())

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
db_ajustado = db.drop(["Pregnancies", "SkinThickness", "BloodPressure"], axis=1)


#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print("Dataframe final: \n", db_ajustado)

print("Distribuição de classes:\n", db_ajustado["Outcome"].value_counts())


#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário


#Salve o dataset atualizado se houver modificações.
db_ajustado.to_csv('db_ajustado.csv')
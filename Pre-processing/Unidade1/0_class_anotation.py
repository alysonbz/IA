from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()
hiking = load_hiking_dataset()
wine = load_wine_dataset()
df1 = load_df1_unidade1()
df2 = load_df2_unidade1()

"""print("Base de dados")
print(df1) # mostra a base de dados
print(volunteer['hits'].dtype)

print("Base de dados")
print(df1)
print("\nCom a funcao info:\n")
print(hiking.info())

print("Base de dados")
print(df1)
print("\nCom a funcao describe:\n")
print(wine.describe())

print("Base de dados")
print(df1)
print("\nCom a funcao dropna:\n")
print(df1.dropna())

print("Base de dados")
print(df1)
print("\nCom a funcao drop:\n")
print(df1.drop([1, 2, 3]))

print("Base de dados")
print(df1)
print("\nCom a funcao isna:\n")
print(df1.isna().sum())  # isna pergunta onde tem e quantos tem not a number, sum soma
print("\nCom a funcao dropna:\n")
print(df1.dropna(subset=["B"]))  #

print("Base de dados")
print(df1)
print(df1.dropna(thresh=2))

print("Base de dados")
print(df1)
print("\nCom a funcao info:\n")
print(volunteer.info())"""

print("BASE DE DADOS 2")
print(df2)
print(df2.info())
df2["C"] = df2["C"].astype("int64") # converte a coluna C em int
print(df2.dtypes)
from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
# É preciso inserir o comando -> pip install -U scikit-learn para instalar as bibliotecas necessárias

import pandas as pd

volunteer = load_volunteer_dataset()
hiking = load_hiking_dataset()
wine  = load_wine_dataset()
df1 = load_df1_unidade1()
df2 = load_df2_unidade1()


#print(hiking.head())

#print(hiking.info())

#print(wine.describe())


print(df1)
# ==== Retira(dropa) os Nan ´s do dataset ====
#print(df1.dropna())

# ==== Retira(dropa) as colunas [1,2,3] ´s do dataset ====
#print(df1.drop([1,2,3]))

#
#print(df1.drop(["A","B"], axis= 1))
# === Se não é numero ===
#print(df1.isna())

# === Diz qtd de NAN em cada coluna
#print(df1.isna().sum())

# === Retira os
#print(df1.dropna(subset=["B"]))

# === pq tira além do 2 o 3?
print(df1.dropna(thresh=2))










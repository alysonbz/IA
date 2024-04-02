import pandas as pd
from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print("--- DIMENSÃO ---")
print(volunteer.ndim)
print()

#mostre os tipos de dados existentes no dataset
print("--- TIPOS DE DADOS ---")
print(volunteer.dtypes)
print()

#mostre quantos elementos do dataset estão faltando na coluna
print("--- ELEMENTOS NaN ---")
print(volunteer.isna().sum())
print()

# Exclua as colunas Latitude e Longitude de volunteer
print("\n--- TODAS AS COLUNAS ---\n", volunteer.columns)

volunteer_cols = volunteer.drop(["Latitude", "Longitude"], axis=1)

print("\n--- APÓS REMOVER AS COLUNAS ---\n", volunteer_cols.columns)
print()

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
volunteer_cols.dropna(subset=["category_desc"], inplace=True)
print(volunteer_cols.isna().sum())
print()

# Print o shape do subset
print(volunteer.shape)
print("Linhas:", volunteer.shape[0], "\nColunas:", volunteer.shape[-1])



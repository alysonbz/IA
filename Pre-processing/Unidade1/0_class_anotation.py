from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()
hiking = load_hiking_dataset()
wine  = load_wine_dataset()
df1 = load_df1_unidade1()
df2 = load_df2_unidade1()

# hiking = pd.read_json("hiking.json")
print("DF1")
print(hiking.head())
print("\n")

print(hiking.info())
print("\n")

print(wine.describe())
print("\n")

print(df1)
print(df1.dropna())
print("\n")

print(df1)
print(df1.drop([1, 2, 3]))
print("\n")

print(df1)
print(df1.drop("A", axis=1))
print("\n")

print(df1)
print(df1.isna().sum())
print(df1.dropna(subset=["B"]))
print("\n")

print(df1)
print(df1.dropna(thresh=2))
print("\n")

print(df1)
print(volunteer.info())
print("\n")

print("DF2")
print(df2)
print("\n")

print(df2.info())
print("\n")

df2["C"] = df2["C"].astype("int64")
print(df2.dtypes)
print("\n")

from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()
hiking = load_hiking_dataset()
wine  = load_wine_dataset()
df1 = load_df1_unidade1()
df2 = load_df2_unidade1()

## print(hiking.head())
## print(hiking.info())
## print(wine.describe)
## print(df1.dropna())
### print(df1.drop([1,2,3]))   ### dropa as linhas 1,2,3
### print(df1.drop("A", axis=1))  ##dropa o eixo A
### print(df1.isna().sum())   ##Soma os NA de todas as colunas
### print(df1.dropna(subset=["B"]))
### print(df1.dropna(thresh=2))  ## exlcui as linhas com mais de 2 NA

### print(volunteer.info()) #INFORMAÇAÕES DO DATAS SET
### print(df2.info())
### df2["C"] = df2["C"].astype("int64")
### print(df2.dtypes)
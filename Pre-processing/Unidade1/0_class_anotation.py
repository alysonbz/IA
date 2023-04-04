from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()
hiking = load_hiking_dataset()
wine  = load_wine_dataset()
df1 = load_df1_unidade1()
df2 = load_df2_unidade1()


from src.utils import load_hiking_dataset , load_df2_unidade1,load_wine_dataset, load_df1_unidade1, load_volunteer_dataset
import pandas as pd

volunteer = load_volunteer_dataset()
hiking = load_hiking_dataset()
wine  = load_wine_dataset()
df1 = load_df1_unidade1()
df2 = load_df2_unidade1()

#print(wine.describe())
#print(hiking.head())
#print(hiking.info())
#print(df1)
#print(df2)
#print(df1.dropna())
#print(df1.drop([1, 2, 3]))
#print(df1.drop("A", axis=1))
#print(df1.isna().sum())
#print(df1.dropna(subset=["B", "A", "C"]))
#print(df1.dropna(thresh=2))
#print(volunteer.info())
#print(df2)
#print(df2.info())
#df2["3"] = df2["C"].astype("int64")
#print(df2.dtypes)

#questão 1:
#from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

# Mostre a dimensão do dataset volunteer
print(volunteer.shape)

#mostre os tipos de dados existentes no dataset
print(volunteer["locality"].info())

#mostre quantos elementos do dataset estão faltando na coluna
print(volunteer["locality"].isnull().sum())

# Exclua as colunas Latitude e Longitude de volunteer
#volunteer_cols =

# Exclua as linhas com valores null da coluna category_desc de volunteer_cols
#volunteer_subset =

# Print o shape do subset
#___

# questão 2:
#from src.utils import load_volunteer_dataset
volunteer = load_volunteer_dataset()

# Print os primeiros elementos da coluna hits
___

# Print as caracteristicas da coluna hits
__

# Converta a coluna hits para o tipo int
___

# Print as caracteristicas da coluna hits novamente
---

#questão 3:

#from src.utils import load_volunteer_dataset
_____

volunteer = load_volunteer_dataset()

# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = __

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer

# mostre o balanceamento das classes em 'category_desc'
print(___['category_desc'].__,'\n','\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.__(__, axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = __[['__']]

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
X_train, X_test, y_train, y_test = __(__, __, stratify=__, random_state=42)

# mostre o balanceamento das classes em 'category_desc' novamente
___


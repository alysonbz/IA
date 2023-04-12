import pandas as pd
from src.utils import load_volunteer_dataset
from src.utils import load_hiking_dataset
from src.utils import load_wine_dataset
from src.utils import load_df1_unidade1
from src.utils import load_df2_unidade1



volunteer = load_volunteer_dataset()
hiking = load_hiking_dataset()
wine  = load_wine_dataset()
df1 = load_df1_unidade1()
df2 = load_df2_unidade1()


"""print(volunteer['hits'].dtype)"""


#print(df1)

#print(df1.drop("A", axis=1))

#print(df1.isna().sum())

#print(df1.dropna(subset=["B"]))

#print(df1.dropna(thresh=2))

#print(volunteer.info())




#print(df2)

#print(df2.info())


df2["C"] = df2["C"].astype("int64")
print(df2.dtypes)






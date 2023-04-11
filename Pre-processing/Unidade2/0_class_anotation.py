from src.utils import  load_df1_unidade2,load_df2_unidade2
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

df1 = load_df1_unidade2()
df2 = load_df2_unidade2()

print(df1, "\n", "\n")
print(df1.var(), "\n", "\n")

df1["log_2"] = np.log(df1["col2"])
print(df1, "\n", "\n")

print(df1[["col1","log_2"]].var(), "\n", "\n")

print(df2.var())

scaler = StandardScaler()
df2_scaled = pd.DataFrame(scaler.fit_transform(df2), columns= df2.columns)

print(df2_scaled)
print(df2_scaled.var())

from src.utils import  load_df1_unidade2,load_df2_unidade2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df1 = load_df1_unidade2()
df2 = load_df2_unidade2()

#DF1 TESTES - variância

print(df1, '\n')
print(df1.var(), '\n')

df1["log_2"] = np.log(df1["col2"])

print(df1, '\n')
print(df1[["col1", "log_2"]].var(), '\n')

#DF2 TESTES - variância

print(df2,'\n')
print(df2.var(),'\n')

#Normalização escalar

scaler= StandardScaler()
df2_scaled = pd.DataFrame(scaler.fit_transform(df2), columns= df2.columns)

print(df2_scaled, '\n')
print(df2_scaled.var(), '\n')
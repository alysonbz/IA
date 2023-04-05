import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.utils import load_df1_unidade2,load_df2_unidade2

df1 = load_df1_unidade2()
df2 = load_df2_unidade2()

#Codigo de normalização logaritmica
#df1["log_2"] = np.log(df1["col2"])
#print(df1[['col1', 'log_2']].var())
#print(df1)
#print(df1.var())

#Codigo para normalização escalar
scaler = StandardScaler()
df2_scaled = pd.DataFrame(scaler.fit_transform(df2), columns=df2.columns)
print(df2_scaled)
print(df2_scaled.var())

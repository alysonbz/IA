import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils import  load_df1_unidade2,load_df2_unidade2

df1 = load_df1_unidade2()
df2 = load_df2_unidade2()

df1["log_2"] = np.log(df1["col2"])
#print(df1)

#print(df1[["col1", "log_2"]].var())

#print(df1)

#print(df1.var())

# Faz com que  a média ocile entre 1 0 -1
scaler = StandardScaler()
df_scaler = pd.DataFrame(scaler.fit_transform(df2))
df_scaler = pd.DataFrame(scaler.fit_transform(df2),
                         columns=df2.columns)
print(df2)




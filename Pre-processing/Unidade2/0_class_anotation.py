import pandas as pd

from src.utils import  load_df1_unidade2,load_df2_unidade2

df1 = load_df1_unidade2()
df2 = load_df2_unidade2()

import numpy as np

df1['Log_2'] = np.log(df1['col2']) # normalização logarítmica
print(df1)

print(df1[['col1', 'Log_2']].var())

print(df2.var())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df2), # normalização escalar
                         columns= df2.columns)

print(df_scaled.var())


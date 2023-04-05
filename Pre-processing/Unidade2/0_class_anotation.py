import numpy as np
from  sklearn.preprocessing import StandardScaler

from src.utils import load_df1_unidade2,load_df2_unidade2

scaler = StandardScaler()
df1 = load_df1_unidade2()
df2 = load_df2_unidade2()
df2_scaled = (scaler.fit_transform(df2))
#print(df1)
#print(df1.var())
#df1['log_2'] = np.log(df1['col1'])
#print(df1)
#print(df1[['col1', 'log_2']].var())
#print(df2)
#print(df2.var())
#print(df2_scaled.var())

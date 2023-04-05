from src.utils import load_df1_unidade2,load_df2_unidade2

df1 = load_df1_unidade2()
df2 = load_df2_unidade2()

print(df1)

import numpy as np
df1["log_2"] = np.log(df1["col2"])
print(df1)


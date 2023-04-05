import numpy as np
from src.utils import load_wine_dataset
import pandas as pd

wine = load_wine_dataset()

pd.set_option('display.max_columns', None)

#print as caractéristicas estatísticas do dataset wine
print("Caractéristicas estatísticas: \n", wine.describe(), '\n')

## Aplique a função de nomarlização logarítmica na coluna Proline
wine['Proline_log'] = np.log(wine['Proline'])
print("Nomarlização logarítmica na coluna Proline: \n",wine, '\n')

# Print a variância da coluna proline
print("Var Proline:", np.var(wine['Proline']),'\n')

# print a variância da coluna proline normalizada
print("Var Proline_log:",np.var(wine['Proline_log']))
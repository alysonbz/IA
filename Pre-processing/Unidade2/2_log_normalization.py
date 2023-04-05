import numpy as np
from src.utils import load_wine_dataset
import pandas as pd

wine = load_wine_dataset()

pd.set_option('display.max_columns', None)

#print as características estatísticas do dataset wine
print('\nCaracterísticas estatísticas do dataset wine')
print(wine.describe())

## Aplique a função de normalização logarítmica na coluna Proline
print('\nFunção de normalização logarítmica na coluna Proline')
wine['Proline_log'] = np.log(wine['Proline'])

# Print a variância da coluna proline
print('\nVariância da coluna proline')
print(np.var(wine['Proline']))

# print a variância da coluna proline normalizada
print('\nVariância da coluna proline normalizada')
print(np.var(wine['Proline_log']))

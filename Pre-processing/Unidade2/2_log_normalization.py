import numpy as np
from src.utils import load_wine_dataset
import pandas as pd

wine = load_wine_dataset()

pd.set_option('display.max_columns', None)

print('print as caractéristicas estatísticas do dataset wine')
print(wine.describe())

print('Aplique a função de nomarlização logarítmica na coluna Proline')
wine['Proline_log'] = np.log(wine['Proline'])

print('Print a variância da coluna proline')
#sem log
print(np.var(wine['Proline']))

print('print a variância da coluna proline normalizada')
#com logaritmo
print(np.var(wine['Proline_log']))
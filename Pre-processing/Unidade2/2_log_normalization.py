import numpy as np
from src.utils import load_wine_dataset
import pandas as pd

wine = load_wine_dataset()

pd.set_option('display.max_columns', None)

#print as caractéristicas estatísticas do dataset wine
print("caractéristicas estatísticas do dataset wine\n", wine.describe())

## Aplique a função de nomarlização logarítmica na coluna Proline
wine["Proline_log"] = np.log(wine["Proline"])
#
# Print a variância da coluna proline
print("variância da coluna proline\n", wine.var())

# print a variância da coluna proline normalizada
print("Variância da coluna proline normalizada\n", np.var(wine['Proline_log']))
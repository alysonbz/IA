# Importar as bibliotecas necessárias

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.model_selection import train_test_split

#Carregar o conjunto de dados

lol = pd.read_csv(r'C:\Users\eryka\Downloads\Master_Ranked_Games.csv\Master_Ranked_Games.csv')
print(lol.info)
print(lol.columns)


##import linkage and dendogram

from scipy.cluster.hierarchy import dendrogram, linkage

X_train, samples, y_train, varieties = lol

# Calcular o linkage
mergings = linkage(lol, method='complete')


# Plotar o dendrograma
dendrogram(mergings,
           labels='gameId',
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()

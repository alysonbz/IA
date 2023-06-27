'''Faça uma análise do dataset utilizando dendograma. Verifique as possibilidades de clusterização e aplique o k-medias.
Observe os resultados e descreva sua iterpretação no relatório. Dica: Observe se há necessidade de normalização dos dados
nas colunas ou nas linhas.
'''

import pandas as pd
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


# Lendo a base de dados
financial_distress = pd.read_csv("Financial Distress.csv")
pd.set_option('display.max_columns', None)
print("\n Dataset: Dificuldade financeira")
print(financial_distress)

# Verificando se existe células vazias ou Nan
print("\nVerificando se existe células vazias ou Nan")
print(financial_distress.isna().sum())


# Separando X e y
X_train, samples, y_train, varieties = load_grains_splited_datadet()


# Implementando o dendograma
mergings = linkage( , method= 'complete')
dendrogram(mergings,
           labels = ,
           leaf_rotation = 90,
           leaf_font_size= 6)
plt.show()






# Separando X e y
'''
X = financial_distress[""].values.reshape(-1, 1) # marca
y = financial_distress["Financial Distress"].values # preço

# Implementando o PCA
model = PCA()
model.fit()

transform'''
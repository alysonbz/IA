'''Faça uma análise do dataset utilizando dendograma. Verifique as possibilidades de
clusterização e aplique o k-medias. Observe os resultados e descreva sua interpretação no
relatório. Dica: Observe se há necessidade de normalização dos dados nas colunas ou nas linhas.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans


# Lendo a base de dados
financial_distress = pd.read_csv("Financial Distress.csv")
pd.set_option('display.max_columns', None)

print("\n Dataset: Dificuldade financeira")
print(financial_distress)

# Transformando a variável alvo "Financial Distress" para 0 e 1, segundo o Kaggle mandou: se for maior que
# -0,50 a empresa deve ser considerada saudável (0). Caso contrário, seria considerada financeiramente
# em dificuldades (1).
financial_distress['Financial Distress'] = financial_distress['Financial Distress'].apply(lambda x: 0 if x > -0.50 else 1)

# Verificando se existem valores vazios ou Nan
print("\nVerificando se existe células vazias ou Nan")
print(financial_distress.isna().sum())

# Excluindo valores vazios
fd_atualizado = financial_distress.dropna()
print(fd_atualizado)

# Fazendo a normalização das linhas
normalized_fd = normalize(financial_distress, axis=1)

# Separando X e y
sem_fd = financial_distress.drop(['Financial Distress'], axis=1)    # X
somente_fd = financial_distress['Financial Distress'].values        # y

# Calculando o linkage: mergings
mergings = linkage(normalized_fd, method='complete')

# Plotando o dendrograma
plt.title("Dendrograma (Agrupamento Hierárquico)")
dendrogram(mergings,
           labels=somente_fd,
           leaf_rotation=90,
           leaf_font_size=6)
plt.show()


# Clusterização K_means
model = KMeans(n_clusters=5)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(sem_fd)

# Create a DataFrame with labels and varieties as columns: df
df_fd = pd.DataFrame({'labels': labels, 'somente_fd': somente_fd})

# Create crosstab: ct
ct = pd.crosstab(df_fd['labels'], df_fd['somente_fd'])

# Display ct
print("\nCrosstab:")
print(ct)


############################################################

# Salvando o dataset normalizado e com o pré processamento
fd_atualizado.to_csv("Financial Distress Atualizado.csv", index=False)
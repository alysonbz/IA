#importe as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

## Carregar o dataset
cogu_df = pd.read_csv('mushrooms.csv')

# Criar uma instância do LabelEncoder
label_encoder = LabelEncoder()

# Percorrer as colunas do dataset
for coluna in cogu_df.columns:
    # Verificar se a coluna contém valores de string
    if cogu_df[coluna].dtype == 'object':
        # Aplicar o LabelEncoder na coluna
        cogu_df[coluna] = label_encoder.fit_transform(cogu_df[coluna])



# Verificar se tem Na ou isnull
'''print(cogu_df.isna().sum())
print(cogu_df.isnull().sum())'''


# DENDOGRAMA
cogu = cogu_df.drop(['class'], axis=1)
clas = cogu_df['class'].values

# Normalizar os dados
normalized_test = normalize(cogu)


# Calcular o linkage: mergings
mergings = linkage(normalized_test, method='complete')
# Plotar o dendrograma
plt.figure(figsize=(10, 6))
dendrogram(mergings,
           labels=clas,
           leaf_rotation=90,
           leaf_font_size=8)

plt.title('Dendrograma')
plt.xlabel('Amostras')
plt.ylabel('Distância')
plt.show()


# clusterização e aplique o k-medias
model = KMeans(n_clusters=3)

# Use fit_predict em model
labels = model.fit_predict(cogu)
# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'Area': clas})
# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['Area'])
# Display ct
print(ct)

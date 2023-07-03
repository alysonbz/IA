#importe as bibliotecas necessárias
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


## Carregar o dataset
oleo_df = pd.read_csv(r"C:\Users\UFC\Downloads\savio\IA\AV3\oil_spill.csv")


# DENDOGRAMA
test = oleo_df.drop(['target'], axis=1)
Area = oleo_df['target'].values

# Normalizar os dados
normalized_test = normalize(test)

# Calcular o linkage: mergings
mergings = linkage(normalized_test, method='complete')
# Plotar o dendrograma
plt.figure(figsize=(10, 6))
dendrogram(mergings,
           labels=Area,
           leaf_rotation=90,
           leaf_font_size=8)

plt.title('Dendrograma')
plt.xlabel('Amostras')
plt.ylabel('Distância')
plt.show()


# clusterização e aplique o k-medias
model = KMeans(n_clusters=4)

# Use fit_predict em model
labels = model.fit_predict(test)
# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'Area': Area})
# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['Area'])
# Display ct
print(ct)



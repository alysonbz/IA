import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans

# Carregar o dataset
train_df = pd.read_csv('train_dataset.csv')

# Dendograma
train = train_df.drop(['Class'], axis=1)
Class = train_df['Class'].values

# Normalizar os dados
normalized_train = normalize(train)

# Calcular o linkage: mergings
mergings = linkage(normalized_train, method='complete')

# Plotar o dendrograma
plt.figure(figsize=(10, 6))
dendrogram(mergings,
           labels=Class,
           leaf_rotation=90,
           leaf_font_size=8)

plt.title('Dendrograma')
plt.xlabel('Amostras')
plt.ylabel('Distância')
plt.show()

# clusterização e aplique o k-medias
model = KMeans(n_clusters=3)

# Use fit_predict em model
labels = model.fit_predict(train)

# Criando um dataframe
df = pd.DataFrame({'labels': labels, 'Class': Class})

# Criação do crosstab: ct
ct = pd.crosstab(df['labels'], df['Class'])

# Display ct
print(ct)

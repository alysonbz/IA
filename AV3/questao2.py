import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Carregue o conjunto de dados em um DataFrame do Pandas

df = pd.read_csv('oil_spill.csv')

# Redução de dimensionalidade usando T-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(df)

# Redução de dimensionalidade usando PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df)

# Plote os gráficos
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=df['target'])
plt.title('T-SNE')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')

plt.subplot(1, 2, 2)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df['target'])
plt.title('PCA')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')

plt.tight_layout()
plt.show()


import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

estado_do_olho0 = pd.read_csv(r"C:\Users\LAB1_00\Desktop\BIANCA\ia_novo\IA\AV3\archive (2)\EEG_Eye_State_Classification.csv")
estado_do_olho2 = estado_do_olho0.sample(n=1498, replace=False)

# Redução de dimensionalidade usando T-SNE
tsne = TSNE(n_components=2, random_state=0)
tsne_result = tsne.fit_transform(estado_do_olho2)

# Redução de dimensionalidade usando PCA
pca = PCA(n_components=2, random_state=0)
pca_result = pca.fit_transform(estado_do_olho2)

# Plotar os resultados do T-SNE
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
plt.title('T-SNE')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')

# Plotar os resultados do PCA
plt.subplot(1, 2, 2)
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title('PCA')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')

plt.tight_layout()
plt.show()
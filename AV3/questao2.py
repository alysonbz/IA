import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

glass_df = pd.read_csv("glass.csv")
glass_df.info()

numeric_columns = glass_df.drop(['Type'], axis=1)

tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(numeric_columns)

pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(numeric_columns)

plt.figure(figsize=(12, 5))


plt.subplot(1, 2, 1)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=glass_df['Type'])
plt.title('T-SNE')

plt.subplot(1, 2, 2)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=glass_df['Type'])
plt.title('PCA')
plt.tight_layout()
plt.show()
pca_result_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_result_df.to_csv("pcs_result.csv", index=False)
tsne_result_df = pd.DataFrame(tsne_result, columns=['Dim1', 'Dim2'])
tsne_result_df.to_csv("tsne_result.csv", index=False)
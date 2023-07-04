import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr



dataset = pd.read_csv(r"C:\Users\Aluno\Documents\Thais\IA\AV3\archive (1)\cancer_classification.csv")
X = dataset.drop(['benign_0__mal_1'], axis=1)
y = dataset['benign_0__mal_1'].values
#z = normalized_dataset = normalize(X)


#tsne_features = model.fit_transform(X)

#xs = tsne_features[:, 0]
#ys = tsne_features[:, 1]

#plt.scatter(xs, ys, alpha = 0.5)

#for x, y, benign_0__mal_1 in zip(xs, ys, y):
 #   plt.annotate(benign_0__mal_1, (x, y), fontsize=5, alpha=0.75)
#plt.show()

model = TSNE(n_components=2, random_state=42)
X_tsne = model.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.title('T-SNE')
print(model)
print(X_tsne)

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.title('PCA')
print(pca)
print(X_pca)
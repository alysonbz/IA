import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

dataset = pd.read_csv(r"C:\Users\Aluno\Documents\Thais\IA\AV3\archive (1)\cancer_classification.csv")
X = dataset.drop(['benign_0__mal_1', 'smoothness error'], axis=1)
y = dataset['benign_0__mal_1']

model = TSNE(n_components=2, random_state=42)
X_tsne = model.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.title('T-SNE')

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.title('PCA')


X_pca_train, X_pca_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
X_tsne_train, X_tsne_test, y_train, y_test = train_test_split(X_tsne, y, test_size=0.2, random_state=42)

# Classificação com os dados reduzidos pelo PCA
classifier_pca = LogisticRegression()
classifier_pca.fit(X_pca_train, y_train)
y_pred_pca = classifier_pca.predict(X_pca_test)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

# Classificação com os dados reduzidos pelo T-SNE
classifier_tsne = LogisticRegression()
classifier_tsne.fit(X_tsne_train, y_train)
y_pred_tsne = classifier_tsne.predict(X_tsne_test)
accuracy_tsne = accuracy_score(y_test, y_pred_tsne)

print("Acurácia com dados reduzidos pelo PCA:", accuracy_pca)
print("Acurácia com dados reduzidos pelo T-SNE:", accuracy_tsne)

variances = X.var()
sorted_columns = variances.sort_values(ascending=False).index

num_columns_to_keep = 2

X_reduced = X[sorted_columns[:num_columns_to_keep]]

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)


classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


print("Acurácia antes da redução de dimensionalidade:", accuracy_pca)
print("Acurácia após a redução de dimensionalidade:", accuracy)

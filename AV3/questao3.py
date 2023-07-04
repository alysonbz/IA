from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

glass_df = pd.read_csv("glass.csv")

y = glass_df['Type']

X = glass_df.drop(['Type'], axis=1)

tsne = TSNE(n_components=2)
X_reduced_tsne = tsne.fit_transform(X)

pca = PCA(n_components=2)
X_reduced_pca = pca.fit_transform(X)

X_reduced = np.concatenate((X_reduced_pca, X_reduced_tsne), axis=1)

y = glass_df['Type']

X = glass_df.drop(['Type'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)


knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)
y_pred_encoded = label_encoder.transform(y_pred)

precision = precision_score(y_test_encoded, y_pred_encoded, average='weighted')
recall = recall_score(y_test_encoded, y_pred_encoded, average='weighted')

print("Precisão:", precision)
print("Recall:", recall)
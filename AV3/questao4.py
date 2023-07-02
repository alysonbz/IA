from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd


cancer_df = pd.read_csv("breast-cancer.csv")


X = cancer_df.drop(['id', 'diagnosis'], axis=1)
y = cancer_df['diagnosis']

pca = PCA()
X_reduced = pca.fit_transform(X)

# Determine as colunas de maior variância
variances = pca.explained_variance_ratio_
variances_sorted = sorted(variances, reverse=True)
threshold = 0.01  # Defina um limiar para a variância

# Obtenha o índice das colunas com maior variância
columns_to_keep = []
for i, variance in enumerate(variances):
    if variance >= threshold:
        columns_to_keep.append(i)

# Selecione apenas as colunas com maior variância
X_reduced = X_reduced[:, columns_to_keep]


X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)

# Mapeie os rótulos 'B' e 'M' para 0 e 1
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)
y_pred_encoded = label_encoder.transform(y_pred)

precision = precision_score(y_test_encoded, y_pred_encoded)
recall = recall_score(y_test_encoded, y_pred_encoded)

print("Precisão:", precision)
print("Recall:", recall)

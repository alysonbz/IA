from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import LabelEncoder


#Parte 1
cancer_df = pd.read_csv("breast-cancer.csv")

# Extrair coluna de classes
y = cancer_df['diagnosis']

# Remover colunas 'id' e 'diagnosis' do dataframe original
X = cancer_df.drop(['id', 'diagnosis'], axis=1)

tsne = TSNE(n_components=2)
X_reduced_tsne = tsne.fit_transform(X)

pca = PCA(n_components=2)
X_reduced_pca = pca.fit_transform(X)

X_reduced = np.concatenate((X_reduced_pca, X_reduced_tsne), axis=1)


#Parte 2

# Extrair coluna de classes
y = cancer_df['diagnosis']

# Remover colunas 'id' e 'diagnosis' do dataframe original
X = cancer_df.drop(['id', 'diagnosis'], axis=1)

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)




#Parte 3
# Crie uma instância do classificador KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Treine o modelo com os dados de treinamento
knn.fit(X_train, y_train)

#Parte 4

y_pred = knn.predict(X_test)

# Parte 5

# Mapeie os rótulos 'B' e 'M' para 0 e 1
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)
y_pred_encoded = label_encoder.transform(y_pred)


# Calcule a precisão e o recall
# Calcule a precisão e o recall
precision = precision_score(y_test_encoded, y_pred_encoded)
recall = recall_score(y_test_encoded, y_pred_encoded)

print("Precisão:", precision)
print("Recall:", recall)
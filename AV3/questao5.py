import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#importando dados
df = pd.read_csv('oil_spill.csv')


X = df.drop('target', axis=1)  # Features (colunas f_1 a f_49)
y = df['target']  # Target (coluna target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Pré-processamento com PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=10)  # Defina o número de componentes desejado
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Classificador KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)
y_pred_knn = knn.predict(X_test_pca)

# Métricas de avaliação para KNN
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)

# Classificador Regressão Logística
logreg = LogisticRegression()
logreg.fit(X_train_pca, y_train)
y_pred_logreg = logreg.predict(X_test_pca)

# Métricas de avaliação para Regressão Logística
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
precision_logreg = precision_score(y_test, y_pred_logreg)
recall_logreg = recall_score(y_test, y_pred_logreg)
f1_logreg = f1_score(y_test, y_pred_logreg)

# Comparação dos resultados
print("Resultados do KNN:")
print("Acurácia:", accuracy_knn)
print("Precisão:", precision_knn)
print("Recall:", recall_knn)
print("F1-score:", f1_knn)
print()

print("Resultados da Regressão Logística:")
print("Acurácia:", accuracy_logreg)
print("Precisão:", precision_logreg)
print("Recall:", recall_logreg)
print("F1-score:", f1_logreg)

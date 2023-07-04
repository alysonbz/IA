import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.utils import load_fish_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# lendo o dataset
samples = load_fish_dataset()

# Codificar a variável igual a Thaís
lb = LabelEncoder()
especie_encoded = lb.fit_transform(samples["specie"])

samples = samples.drop(['specie'], axis=1)

# Padronizar os atributos
scaler = StandardScaler()
scaled_samples = scaler.fit_transform(samples)

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(scaled_samples, especie_encoded, test_size=0.2, random_state=42)

# instânciar o PCA
pca = PCA(n_components=2)

pca.fit(X_train)

# Transformar os dados de treinamento e teste com PCA
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# usar o classificador do Henrique
clf = RandomForestClassifier()
clf.fit(X_train_pca, y_train)

# Predict
y_pred = clf.predict(X_test_pca)

accuracy = accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)

# classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

#matriz de confusão
confusion_mat = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(confusion_mat)

plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred)
plt.title("Scatter Plot - PCA")
plt.show()

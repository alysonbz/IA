import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Carregar o dataset do arquivo CSV
lol_sample = pd.read_csv('lol_sample.csv')

# Carregar o dataset do arquivo CSV
lol_sample = pd.read_csv('C:/Users/eryka/OneDrive/Área de Trabalho/444/IA/AV3/lol_sample.csv')


# Separar os atributos das classes
X = lol_sample.drop('blueWins', axis=1)
y = lol_sample['blueWins']

# Normalizar os atributos
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Redução de dimensionalidade com T-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_normalized)

# Redução de dimensionalidade com PCA
pca = PCA(n_components=2)
pca.fit(X_normalized)
X_pca = pca.transform(X_normalized)

# Dividir os dados reduzidos em conjuntos de treinamento e teste
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
X_train_tsne, X_test_tsne, y_train, y_test = train_test_split(X_tsne, y, test_size=0.2, random_state=42)

# Criar uma instância do classificador SVM
svm = SVC()

# Treinar o modelo utilizando os dados reduzidos por PCA
svm.fit(X_train_pca, y_train)

# Realizar a classificação utilizando os dados de teste reduzidos por PCA
y_pred_pca = svm.predict(X_test_pca)

# Calcular a acurácia do modelo utilizando os dados reduzidos por PCA
accuracy_pca = accuracy_score(y_test, y_pred_pca)

# Treinar o modelo utilizando os dados reduzidos por T-SNE
svm.fit(X_train_tsne, y_train)

# Realizar a classificação utilizando os dados de teste reduzidos por T-SNE
y_pred_tsne = svm.predict(X_test_tsne)

# Calcular a acurácia do modelo utilizando os dados reduzidos por T-SNE
accuracy_tsne = accuracy_score(y_test, y_pred_tsne)

# Comparar as acurácias obtidas por PCA e T-SNE
print("Acurácia utilizando PCA:", accuracy_pca)
print("Acurácia utilizando T-SNE:", accuracy_tsne)

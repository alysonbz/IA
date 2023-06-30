#bibliotecas relevantes
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Carregar o dataset do arquivo CSV
lol_sample = pd.read_csv('lol_sample.csv')

# Carregar o dataset do arquivo CSV
lol_sample = pd.read_csv('C:/Users/eryka/OneDrive/Área de Trabalho/444/IA/AV3/lol_sample.csv')

# Definir as variáveis de entrada (X) e a variável de saída (y)
X = lol_sample.drop('blueWins', axis=1)
y = lol_sample['blueWins']

# Normalizar os atributos
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Criar uma instância do modelo PCA com o número adequado de componentes
pca = PCA(n_components=2)

# Ajustar o modelo PCA aos dados padronizados
pca.fit(X_normalized)

# Transformar os dados padronizados: pca_features
pca_features = pca.transform(X_normalized)

# Visualizar o gráfico de dispersão com a redução de dimensionalidade
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=y)
plt.xlabel('Dimensão 1')
plt.ylabel('Dimensão 2')
plt.title('Redução com PCA')
plt.show()


# Redução de dimensionalidade com T-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_normalized)

# Plotar gráfico T-SNE
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.title('Redução com T-SNE')
plt.xlabel('Dimensão 1')
plt.ylabel('Dimensão 2')
plt.show()



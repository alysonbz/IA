
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from questao1 import database

X = database.drop(['CLASS'], axis=1)
y = database['CLASS'].values

#INICIALIZANDO
scaler = StandardScaler()
lb = LabelEncoder()

# T-SNE
normalized = normalize(X)

# Crie uma instância TSNE: model
model = TSNE(n_components=2)

# Aplique fit_transform às amostras: tsne_features
tsne_features = model.fit_transform(normalized)

# Selecione o 0º recurso: xs
xs = tsne_features[:,0]

# Selecione o 1º recurso: ys
ys = tsne_features[:,1]

# Gráfico de dispersão, colorindo por variedade_números
plt.scatter(xs, ys, c=y)
plt.show()

# PCA
scaled_samples = scaler.fit_transform(X)
class_pca = lb.fit_transform(database['CLASS'])

# Crie um modelo PCA com 2 componentes
pca = PCA(n_components=2)

# Ajustar a instância do PCA às amostras dimensionadas
pca.fit(scaled_samples)

# Transforme as amostras dimensionadas: pca_features
transformed = pca.transform(scaled_samples)

# Visualize o gráfico de dispersão com dimensão reduzida
xs = transformed[:, 0]
ys = transformed[:, 1]
plt.scatter(xs, ys, c=class_pca)
plt.title('PCA')
plt.show()
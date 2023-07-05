import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# INICIALIZANDO
scaler = StandardScaler()
label_encoder = LabelEncoder()

# PROCESSANDO
train_df = pd.read_csv('train_dataset.csv')

train = train_df.drop(['Class'], axis=1)
Class = label_encoder.fit_transform(train_df['Class'])

# T-SNE
normalized_train = normalize(train)

# Instancializar o TSNE em model
model = TSNE(n_components=2)

# Aplicar o fit transform no test normalizado
tsne = model.fit_transform(normalized_train)

# Select the 0th feature: xs
xs = tsne[:,0]
# Select the 1st feature: ys
ys = tsne[:,1]
# Scatter plot, Class
plt.scatter(xs, ys, c=Class)
plt.show()

# PCA
scaled_train = scaler.fit_transform(train)

# Criar um PCA
pca = PCA(n_components=2)

# Fit o PCA em scaled_train
pca.fit(scaled_train)

# Transforme o scaled_train
transformed = pca.transform(train)

# Gráfico
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=Class)
plt.show()

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

#PROCESSANDO
cogu_df = pd.read_csv('mushrooms.csv')

# Criar uma instância do LabelEncoder
label_encoder = LabelEncoder()

# Percorrer as colunas do dataset
for coluna in cogu_df.columns:
    # Verificar se a coluna contém valores de string
    if cogu_df[coluna].dtype == 'object':
        # Aplicar o LabelEncoder na coluna
        cogu_df[coluna] = label_encoder.fit_transform(cogu_df[coluna])



cogu = cogu_df.drop(['class'],axis=1)
clas = cogu_df['class'].values

#INICIALIZANDO
scaler = StandardScaler()


# T-SNE
normalized_cogu = normalize(cogu)

# Instancializar o TSNE em model
model = TSNE(n_components=2)

# Aplicar o fit transform no test normalizado
tsne = model.fit_transform(normalized_cogu)

# Select the 0th feature: xs
xs = tsne[:,0]
# Select the 1st feature: ys
ys = tsne[:,1]
# Scatter plot, Area
plt.scatter(xs, ys, c=clas)
plt.show()


#PCA
scaled_cogu = scaler.fit_transform(cogu)

# Criar um PCA
pca = PCA(n_components=2)

# Fit o PCA em scaled_test
pca.fit(scaled_cogu)

# Transform o scaled_test: transformed
transformed = pca.transform(cogu)

#Gráfico
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=clas)
plt.show()
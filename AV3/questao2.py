# 1 - Reduza o dataset com o T-SNE para duas dimensões e plote o gráfico do atributos que a técnicas gerou.
# 2 - Reduza o dataset com o PCA para duas dimensões e plote o gráfico do atributos que a técnicas gerou.
# De forma subjetiva e visual, qual dos gráficos você acredita que vai possuir um
# melhor desempenho em um processo de classificação utilizando os dois atribuitos?

# pacotes
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# dados
df = pd.read_csv(r'C:\Users\guilh\OneDrive\Documentos\GD\INCART 2-lead Arrhythmia Database.csv')
df = df.dropna()

# converter valores categóricos em valores numéricos
encoder = LabelEncoder()
df['type'] = encoder.fit_transform(df['type'])

sample_size = int(0.3 * len(df))
random_sample = df.sample(n=sample_size, random_state=42)

samples = random_sample.drop(['type', 'record'], axis=1).select_dtypes(exclude="object")
types = random_sample['type'].values

normalized = normalize(samples)

#1
# instanciar o TSNE
tmodel = TSNE(n_components=2)

# aplicar o fit_transform
tsne_features = tmodel.fit_transform(normalized)

# feature 0: txs, feature 1: tys
txs = tsne_features[:,0]
tys = tsne_features[:,1]

# printar o shape (formato) de pca_features
print(tsne_features.shape)

# plotar gráfico de dispersão da dimensão reduzida colorido por types
plt.scatter(txs, tys, c=types)
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.title('t-SNE')
plt.show()

#2
# instanciar o PCA para duas dimensões
pmodel = PCA(n_components=2)

# ajustar
pmodel.fit(normalized)

# transformar
pca_features = pmodel.transform(normalized)

# printar o shape (formato) de pca_features
print(pca_features.shape)

# plotar gráfico de dispersão da dimensão reduzida colorido por types
pxs = pca_features[:,0]
pys = pca_features[:,1]
plt.scatter(pxs, pys, c=types)
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.title('PCA')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Carregar o dataset
df1 = pd.read_csv('mitbih_train.csv')


# Selecionar as colunas relevantes para a normalização
columns = ['1.000000000000000000e+00',
            '9.003241658210754395e-01',
            '3.585899472236633301e-01',
            '5.145867168903350830e-02',
          '4.659643396735191345e-02']  # substituir pelas colunas relevantes do dataset

# Criar um dataframe apenas com as colunas selecionadas
df1_selected = df1[columns]

# Aplicar t-SNE para reduzir para duas dimensões
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(df1)

# Aplicar PCA para reduzir para duas dimensões
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(df1)

# Plotar gráfico do t-SNE
plt.figure(figsize=(10, 5))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
plt.title("t-SNE")
plt.xlabel("Dimensão 1")
plt.ylabel("Dimensão 2")
plt.show()

# Plotar gráfico do PCA
plt.figure(figsize=(10, 5))
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title("PCA")
plt.xlabel("Dimensão 1")
plt.ylabel("Dimensão 2")
plt.show()

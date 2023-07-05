'''
Reduza o dataset T-SNE e com PCA para duas dimensões. Plote o gráfico do atributos que as duas
técnicas geraram. De forma subjetiva e visual, qual dos gráficos você acredita que vai possuir
um melhor desempenho em um processo de classificação utilizando os dois atribuitos ?
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

financial_distress = pd.read_csv("Financial Distress Atualizado.csv")

# Separando X e y
sem_fd = financial_distress.drop(['Financial Distress'], axis=1)    # X
somente_fd = financial_distress['Financial Distress'].values        # y

# Fazendo a normalização das linhas
normalized_sem_fd = normalize(sem_fd, axis=1)
normalized_fd = normalize(financial_distress.drop(['Financial Distress'], axis=1))


# TSNE
model = TSNE(learning_rate=200)

# Aplicando fit_transform
tsne_features = model.fit_transform(normalized_sem_fd)

# Selecione o recurso 0: xs
xs = tsne_features[:,0]

# Selecione o recurso 1: ys
ys = tsne_features[:,1]

# Plotando o gráfico
plt.scatter(xs, ys, alpha=0.5, c=somente_fd)
plt.title("TSNE")
plt.show()

# Criando o modelo PCA com 2 componentes
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled financial_distress
pca.fit(normalized_fd)

# Transform the scaled financial_distress: pca_features
transformed = pca.transform(normalized_fd)

# Print the shape of pca_features (dados transformados após a aplicação do PCA)
print(transformed.shape)

# Visualize scatter plot with dimension reduced
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=somente_fd)
plt.title("PCA")
plt.show()

# Calculando a coorelação de Pearson de xs e ys
correlation, pvalue = pearsonr(xs, ys)

# Mostrando a correlação
print('\nCorrelação:', correlation)
#print('\nP valor:', pvalue)


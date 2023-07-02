import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#PROCESSANDO
teste_df = pd.read_csv('test_dataset.csv')
test = teste_df.drop(['Area'],axis=1)
Area = teste_df['Area'].values

#INICIALIZANDO
scaler = StandardScaler()


# T-SNE
normalized_test = normalize(test)

# Instancializar o TSNE em model
model = TSNE(n_components=2)

# Aplicar o fit transform no test normalizado
tsne = model.fit_transform(normalized_test)

# Select the 0th feature: xs
xs = tsne[:,0]
# Select the 1st feature: ys
ys = tsne[:,1]
# Scatter plot, Area
plt.scatter(xs, ys, c=Area)
plt.show()



#PCA
scaled_test = scaler.fit_transform(test)

# Criar um PCA
pca = PCA(n_components=2)

# Fit o PCA em scaled_test
pca.fit(scaled_test)

# Transform o scaled_test: transformed
transformed = pca.transform(test)

#Gráfico
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=Area)
plt.show()
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('smoke_detection_iot.csv')
X = df.drop(['FireAlarm'], axis=1)
y = df['FireAlarm'].values

#INICIALIZANDO
scaler = StandardScaler()
lb = LabelEncoder()

# T-SNE
normalized = normalize(X)
# Create a TSNE instance: model
model = TSNE(n_components=2)
# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(normalized)
# Select the 0th feature: xs
xs = tsne_features[:,0]
# Select the 1st feature: ys
ys = tsne_features[:,1]
# Scatter plot, coloring by variety_numbers
plt.scatter(xs, ys, c=y)
plt.show()

#PCA
scaled_samples = scaler.fit_transform(y)
class_pca = lb.fit_transform(df['class'])
# Create a PCA model with components in adequate number: pca
pca = PCA(n_components=2)
# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)
# Transform the scaled samples: pca_features
tramsformed = pca.transform(X)
#vizualize scatter plot with dimension reduced
xs = tramsformed[:,0]
ys = tramsformed[:,1]
plt.scatter(xs, ys, c=class_pca)
plt.show()
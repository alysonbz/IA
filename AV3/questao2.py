import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA



smoking = pd.read_csv("smoking_df.csv")
df = smoking.drop(['smoking'],axis=1)
smoking_valor= smoking['smoking'].values
normalized_df = normalize(df)

#INICIALIZANDO
scaler = StandardScaler()
lb = LabelEncoder()

# T-SNE
# Create a TSNE instance: model
model = TSNE(learning_rate=200)
# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(normalized_df)
# Select the 0th feature: xs
xs = tsne_features[:,0]
# Select the 1st feature: ys
ys = tsne_features[:,1]
# Scatter plot, coloring by variety_numbers
plt.scatter(xs, ys, c=smoking_valor)
plt.show()

"""# Create a TSNE instance: model
model = TSNE(learning_rate=50)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(df)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs, ys, c = smoking_valor)
plt.show()"""

#PCA
scaled_samples = scaler.fit_transform(normalized_df)
class_pca = lb.fit_transform(normalized_df['class'])
# Create a PCA model with components in adequate number: pca
pca = PCA(n_components=2)
# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)
# Transform the scaled samples: pca_features
tramsformed = pca.transform(normalized_df)
#vizualize scatter plot with dimension reduced
xs = tramsformed[:,0]
ys = tramsformed[:,1]
plt.scatter(xs, ys, c=class_pca)
plt.show()
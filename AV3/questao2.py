import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


smoking1 = pd.read_csv("smoking_df.csv")
smoking2 = int(len(smoking1) * 0.04)
smoking = smoking1.sample(n=smoking2)
df = smoking.drop(['smoking'],axis=1)
smoking_valor= smoking['smoking'].values
normalized_df = normalize(df)

#INICIALIZANDO
scaler = StandardScaler()
lb = LabelEncoder()

# T-SNE
# Create a TSNE instance: model
model = TSNE(learning_rate=50)
# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(normalized_df)
# Select the 0th feature: xs
xs = tsne_features[:,0]
# Select the 1st feature: ys
ys = tsne_features[:,1]
# Scatter plot, coloring by variety_numbers
plt.scatter(xs, ys, c=smoking_valor)
plt.show()

scaled_df = scaler.fit_transform(normalized_df)


# Create a PCA model with components in adequate number: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(normalized_df)

# Transform the scaled samples: pca_features
transformed = pca.transform(normalized_df)

# Print the shape of pca_features
print(transformed.shape)

#vizualize scatter plot with dimension reduced
xx = transformed[:,0]
yy = transformed[:,1]
plt.scatter(xx, yy, c=smoking_valor)
plt.show()
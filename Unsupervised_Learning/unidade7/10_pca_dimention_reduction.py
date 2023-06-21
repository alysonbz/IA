import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.utils import load_fish_dataset
from sklearn.preprocessing import LabelEncoder

samples = load_fish_dataset()
samples = samples.drop(['specie'],axis=1)
scaler = StandardScaler()
scaled_samples = scaler.fit_transform(samples)

label_encoder = LabelEncoder()

label_encoder.fit_transform(samples['Bream'])
'''
# Create a PCA model with components in adequate number: pca
pca = PCA(n_components=1.6)

# Fit the PCA instance to the scaled samples
pca.fit(samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(samples)

# Print the shape of pca_features
print(pca_features.shape)

#vizualize scatter plot with dimension reduced
xs = pca_features[:,0]
ys = pca_features[:,1]
plt.scatter(xs, ys, c=samples)
plt.show()

'''
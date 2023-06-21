import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.utils import load_fish_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline

samples = load_fish_dataset()
lb = LabelEncoder()
species = lb.fit_transform(samples["specie"])
samples = samples.drop(['specie'],axis=1)
scaler = StandardScaler()
scaled_samples = scaler.fit_transform(samples)


# Create a PCA model with components in adequate number: pca
pca = PCA(n_components= 2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)


# Transform the scaled samples: pca_features
transformed = pca.fit_transform(scaled_samples)
xs = transformed[:,0]
ys = transformed[:,1]

# Print the shape of pca_features
print(xs.shape, ys.shape)



#vizualize scatter plot with dimension reduced
plt.scatter(xs, ys, c= species)
plt.show()

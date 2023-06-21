import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.utils import load_fish_dataset
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
samples = load_fish_dataset()
specie = lb.fit_transform(samples['specie'])
samples = samples.drop(['specie'],axis=1)
scaler = StandardScaler()
scaled_samples = scaler.fit_transform(samples)


# Create a PCA model with components in adequate number: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
tramsformed = pca.transform(samples)

# Print the shape of pca_features
print(tramsformed.shape)

#vizualize scatter plot with dimension reduced
xs = tramsformed[:,0]
ys = tramsformed[:,1]
plt.scatter(xs, ys, c=specie)
plt.show()


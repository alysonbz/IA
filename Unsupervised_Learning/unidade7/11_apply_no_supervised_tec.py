"""from src.utils import load_fish_dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.utils import load_fish_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


fish_df = load_fish_dataset()
fish_df = fish_df.drop(['specie'],axis=1)
scaler = StandardScaler()
scaled_fish = scaler.fit_transform(fish_df)


# Create a PCA model with components in adequate number: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(fish_df)

# Transform the scaled samples: pca_features
transformed = pca.transform(fish_df)

# Print the shape of pca_features
print(transformed.shape)

X = fish_df.drop['specie'].values
y = fish_df['specie'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

knn = KNeighborsClassifier(n_neighbors=6)

# Fit the model to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""

from src.utils import load_fish_dataset
from sklearn.decomposition import PCA
from src.utils import load_fish_dataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold


fish_df = load_fish_dataset()
samples = fish_df.drop(['specie'],axis=1)
pca = PCA(n_components=2)
# Fit the PCA instance to the scaled samples
pca.fit(samples)
# Transform the scaled samples: pca_features
tramsformed = pca.transform(samples)
# Print the shape of pca_features
print(tramsformed.shape)


#Matriz de confusão
X = tramsformed
y = fish_df['specie']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)
knn = KNeighborsClassifier(n_neighbors=6)
# Fit the model to the training data
knn.fit(X_train, y_train)
# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)
# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))

#Classification Report
print(classification_report(y_test, y_pred))
#Acuracia
kf = KFold(n_splits=6, shuffle=True, random_state=5)
knn = KNeighborsClassifier()
cv_scores = cross_val_score(knn, X, y, cv=kf)
print(cv_scores)


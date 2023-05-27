from src.utils import log_reg_diabetes
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from src.utils import load_diabetes_clean_dataset
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
# Import roc_auc_score
from sklearn.metrics import roc_auc_score

y_prob,y_test,y_pred = log_reg_diabetes()

# Calculate roc_auc_score
print(roc_auc_score(y_test, y_prob))

# Calculate the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Calculate the classification report
print(classification_report(y_test, y_pred))

print("MELHOR K\n")

db = load_diabetes_clean_dataset()
X = db[["pregnancies","glucose","diastolic","triceps","insulin","bmi","dpf","age"]].values
y = db["diabetes"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create neighbors
neighbors = np.arange(1, 12)
train_accuracies = {}
test_accuracies = {}
teste = True
for neighbor in neighbors:
    # Set up a KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=neighbor)

    # Fit the model
    knn.fit(X_train, y_train)

    # Compute accuracy
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

print("acuracy on train: ",train_accuracies, '\n',"acuracy on test: ", test_accuracies)

# Add a title
plt.title("KNN: Varying Number of Neighbors")

# Plot training accuracies
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")

# Plot test accuracies
plt.plot(neighbors, test_accuracies.values(), label="Test Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

# Display the plot
plt.show()

print("\nCROSS VALIDATION KNN\n")

# Import the necessary modules
from sklearn.model_selection import cross_val_score, KFold

db = load_diabetes_clean_dataset()
# Create X and y arrays
X = db["dpf"].values.reshape(-1, 1)
y = db["diabetes"].values

#Create a KFold object
kf = KFold(n_splits=6, shuffle=True, random_state=5)

knn = KNeighborsClassifier()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(knn, X, y, cv=kf)

# Print cv_scores
print(cv_scores)

# Print the mean
print(np.mean(cv_scores))

# Print the standard deviation
print(np.std(cv_scores))

print("\nCROSS VALIDATION RL\n")
# Create X and y arrays
X = db["dpf"].values.reshape(-1, 1)
y = db["diabetes"].values

#Create a KFold object
kf = KFold(n_splits=6, shuffle=True, random_state=5)

Logreg = LogisticRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(Logreg, X, y, cv=kf)

# Print cv_scores
print(cv_scores)

# Print the mean
print(np.mean(cv_scores))

# Print the standard deviation
print(np.std(cv_scores))

print("\nCONFUSION MATRIX KNN\n")
#Import confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

X = db.drop(['diabetes'],axis=1)
y = db['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)

knn = KNeighborsClassifier(n_neighbors=6)

# Fit the model to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

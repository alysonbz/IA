from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

# Load the diabetes dataset (example dataset for demonstration)
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Convert to binary classification by setting a threshold
threshold = 150
y_binary = (y > threshold).astype(int)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Create a logistic regression model
log_reg = LogisticRegression()

# Fit the model on the training data
log_reg.fit(X_train, y_train)

# Obtain predicted probabilities and labels
y_prob = log_reg.predict_proba(X_test)[:, 1]  # Probabilities of positive class
y_pred = log_reg.predict(X_test)

# Calculate roc_auc_score
roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
print("ROC AUC Score:", roc_auc)

# Calculate the confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion_mat)

# Calculate the classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

# Load the diabetes dataset (example dataset for demonstration)
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
log_reg = LogisticRegression()

# Fit the model on the training data
log_reg.fit(X_train, y_train)

# Obtain predicted probabilities and labels
y_prob = log_reg.predict_proba(X_test)[:, 1]  # Probabilities of positive class
y_pred = log_reg.predict(X_test)

# Calculate roc_auc_score
roc_auc = roc_auc_score(y_test, y_prob)
print(roc_auc)

# Calculate the confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print(confusion_mat)

# Calculate the classification report
class_report = classification_report(y_test, y_pred)
print(class_report)


from src.utils import log_reg_diabetes
from sklearn.metrics import classification_report, confusion_matrix

# Import roc_auc_score
from sklearn.metrics import roc_auc_score

y_prob, y_test, y_pred = log_reg_diabetes()

# Calculate roc_auc_score
print(roc_auc_score(y_test, y_pred))

# Calculate the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Calculate the classification report
print(classification_report(y_test, y_prob))



from sklearn.metrics import classification_report, confusion_matrix

#KNN

#biblioteca
from src.utils import load_diabetes_clean_dataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
import numpy as np


diabetes_df = load_diabetes_clean_dataset()

#KNN

X = diabetes_df.drop('diabetes', axis=1)
y = diabetes_df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=12)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = knn.score(X_test, y_test)
print('Acurácia:', accuracy)

# Cross-validation with K-Fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(knn, X, y, cv=kfold)
report = classification_report(y_test, y_pred)


print('Scores de validação cruzada:', cv_scores)
print('Média da acurácia da validação cruzada:', cv_scores.mean())
print(confusion_matrix(y_test, y_pred))
print(report)

# Create neighbors
neighbors = np.arange(1, 12)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    # Set up a KNN Classifier
    knn =  KNeighborsClassifier(n_neighbors=neighbor)

    # Fit the model
    knn.fit(X_train, y_train)

    # Compute accuracy
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)

print("acuracy on train: ",train_accuracies, '\n',"acuracy on test: ", test_accuracies)


# Add a title
plt.title("KNN: Varying Number of Neighbors")

# Plot training accuracies
plt.plot(neighbors, train_accuracies.values(), label="training Accuracy")

# Plot test accuracies
plt.plot(neighbors, test_accuracies.values(), label="Test Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

# Display the plot
plt.show()


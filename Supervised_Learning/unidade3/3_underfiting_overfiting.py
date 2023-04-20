from src.utils import load_churn_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

churn_df = load_churn_dataset()
X = churn_df[["account_length", "total_day_charge", "total_eve_charge", "total_night_charge", "total_intl_charge", "number_customer_service_calls"]].values
y = churn_df["churn"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create neighbors
neighbors = np.arange(1, 21)  # Fill in the range of neighbors you want to try
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:
    # Set up a KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=neighbor)

    # Fit the model
    knn.fit(X_train, y_train)

    # Compute accuracy
    train_predictions = knn.predict(X_train)
    test_predictions = knn.predict(X_test)
    train_accuracies[neighbor] = accuracy_score(y_train, train_predictions)
    test_accuracies[neighbor] = accuracy_score(y_test, test_predictions)

print("Accuracy on train: ", train_accuracies)
print("Accuracy on test: ", test_accuracies)

# Add a title
plt.title("KNN Accuracy with Varying Number of Neighbors")

# Plot training accuracies
plt.plot(neighbors, list(train_accuracies.values()), label="Train Accuracy")

# Plot test accuracies
plt.plot(neighbors, list(test_accuracies.values()), label="Test Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

# Display the plot
plt.show()

from src.utils import load_churn_dataset
import numpy as np
from sklearn.neighbors import KNeighborsClassifier  # Importing KNeighborsClassifier from sklearn.neighbors

churn_df = load_churn_dataset()

# Create arrays for the features and the target variable
y = churn_df["Churn"].values  # Assuming "Churn" is the name of the target variable column
X = churn_df[["TotalCharges", "MonthlyCharges"]].values  # Assuming these are the feature columns

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

X_test = np.array([[30.0, 17.5],
                   [107.0, 24.1],
                   [213.0, 10.9]])

# Predict the labels for the X_test
y_pred = knn.predict(X_test)

# Print the predictions for X_test
print("Predictions: {}".format(y_pred))
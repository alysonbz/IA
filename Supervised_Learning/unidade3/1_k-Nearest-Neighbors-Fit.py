from src.utils import load_churn_dataset
import numpy as np

# Importando KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

churn_df = load_churn_dataset()

# Cria matrizes para os recursos e a variável de destino
y = ____["____"].values
X = ____[["____", "____"]].values

# Create a KNN classifier with 6 neighbors
knn =

# Fit the classifier to the data
knn.____(____, ____)

X_test = np.array([[30.0, 17.5],
                  [107.0, 24.1],
                  [213.0, 10.9]])

# Predict the labels for the X_teste
y_pred = __.__(__)

# Print the predictions for X_test
print("Predictions: {}".format(__))
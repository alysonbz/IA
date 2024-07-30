from sklearn.neighbors import KNeighborsClassifier
from src.utils import load_churn_dataset
from sklearn.model_selection import train_test_split

# Import the module (assuming src.utils is correctly configured and load_churn_dataset is accessible)

churn_df = load_churn_dataset()
X = churn_df[["account_length", "total_day_charge", "total_eve_charge", "total_night_charge", "total_intl_charge", "number_customer_service_calls"]].values
y = churn_df["churn"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy on the test set
print("Imprimir a precisão: {:.2f}".format(knn.score(X_test, y_test)))
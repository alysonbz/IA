from src.utils import load_diabetes_clean_dataset
from sklearn.model_selection import train_test_split
#Import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'],axis=1)
y = diabetes_df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify=y)


# Instantiate the model
logreg = LogisticRegression()
knn = KNeighborsClassifier()

# Fit the model
logreg.fit(X_train, y_train)
knn.fit(X_train,y_train)
Y_pred = logreg.predict(X_test)
y_pred = knn.predict(X_test)

# Predict probabilities
y_pred_probs = logreg.predict_proba(X_test)[:, 1]

print("predição da regressão logistica", y_pred_probs[: 16])
print("Predictions Knn: {}".format(y_pred))
print('score', knn.score(X_test, y_test))


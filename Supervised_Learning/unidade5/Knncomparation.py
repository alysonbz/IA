#biblioteca
from src.utils import load_diabetes_clean_dataset
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

diabetes_df = load_diabetes_clean_dataset()
print(diabetes_df)

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




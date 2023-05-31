from src.utils import load_diabetes_clean_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
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
print(y_pred_probs)

#acuracia
acuracy= knn.score(X_test, y_test)
acura = logreg.score(X_test,y_test)
print("acuracia com Knn:\n", acuracy)
print("acuracia com R.L.:\n", acura)


print("predição da regressão logistica", Y_pred[: 16])
print("Predictions Knn: {}".format(y_pred[: 16]))
print(confusion_matrix, y_test, y_pred)# matriz confusão feita com Knn
print(classification_report(y_test, y_pred))
print(confusion_matrix, y_test, Y_pred)#matriz confusão feita com regressão logistica
print(classification_report(y_test, Y_pred))
#print('score', knn.score(X_test, y_test))
#print("score", logreg.score()
plt.plot(acura, acuracy, color = "blue")
plt.xlabel('m')
plt.ylabel('a')
plt.title('comparação ')
plt.show()
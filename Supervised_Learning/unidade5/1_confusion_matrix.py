from src.utils import load_diabetes_clean_dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Importa a matriz de confusão.
from sklearn.metrics import confusion_matrix, classification_report

diabetes_df = load_diabetes_clean_dataset()
X = diabetes_df.drop(['diabetes'], axis=1)
y = diabetes_df['diabetes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=6)

# Ajusta o modelo aos dados de treinamento
knn.fit(X_train, y_train)

# Preveja os rótulos dos dados de teste: y_pred.
y_pred = knn.predict(X_test)

# Gere a matriz de confusão e o relatório de classificação.
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
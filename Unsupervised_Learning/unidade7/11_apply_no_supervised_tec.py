from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils import load_fish_dataset
fish_df = load_fish_dataset()


# Faça uma redução da dimensão com PCA do dataset
# e escolha um método para classificar os atributos gerados pelo PCA obstidos na questão 10.
# Calcule a acurácia, gere o classification report e a matriz de confusão.

samples = fish_df.drop(['specie'],axis=1)
scaler = StandardScaler()
scaled_samples = scaler.fit_transform(samples)

pca = PCA(n_components=2)
pca.fit(scaled_samples)
transformed = pca.transform(scaled_samples)
print(transformed.shape)

X = transformed
y = fish_df['specie']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


print("Acurácia: ", accuracy_score(y_test, y_pred))
print("Matriz de confusão: ", confusion_matrix(y_test, y_pred))
print("Classification report: ", classification_report(y_test, y_pred))
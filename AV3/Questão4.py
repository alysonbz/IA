import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Carregar o dataset....
df = pd.read_csv("mitbih_train.csv")

# Separar os atributos (X) e os rótulos (y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Aplicar PCA para seleção das colunas de maior variância
pca = PCA(n_components=10)  # Defina o número de componentes desejado
X_pca = pca.fit_transform(X)

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Inicializar o classificador (por exemplo, Regressão Logística)
classifier = LogisticRegression()

# Treinar o classificador
classifier.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = classifier.predict(X_test)

# Calcular a acurácia do classificador
accuracy = (accuracy_score)(y_test,y_pred)
print("Acurácia do classificador:{:.2f}%".format(accuracy * 100))

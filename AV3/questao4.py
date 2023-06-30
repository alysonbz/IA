import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Carregar o conjunto de dados tratado anteriormente
lol = pd.read_csv(r'C:\Users\eryka\Downloads\Master_Ranked_Games.csv\Master_Ranked_Games.csv')

# Realizar a amostragem aleatória
sample_size = 1000
lol_sample = lol.sample(n=sample_size, random_state=42)

# Separar os atributos das classes
X = lol_sample.drop('blueWins', axis=1)
y = lol_sample['blueWins']

# Normalizar os atributos
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Redução de dimensionalidade com PCA
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_normalized)

# Dividir os dados reduzidos em conjuntos de treinamento e teste
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Criar uma instância do classificador SVM
svm = SVC()

# Treinar o modelo utilizando os dados reduzidos por PCA
svm.fit(X_train_pca, y_train)

# Realizar a classificação utilizando os dados de teste reduzidos por PCA
y_pred_pca = svm.predict(X_test_pca)

# Calcular a acurácia do modelo utilizando os dados reduzidos por PCA
accuracy_pca = accuracy_score(y_test, y_pred_pca)

# Imprimir a acurácia do modelo
print("Acurácia utilizando PCA:", accuracy_pca)

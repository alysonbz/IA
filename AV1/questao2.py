# Importe as bibliotecas necessárias.
from questao1 import customer_ajustado
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", message="^In SciPy 1.11.0, the default value of `keepdims` will become False.*$")

#Esse é apenas um aviso do Sklearn, indicando que o comportamento padrão da função modeserá alterado em uma versão futura do SciPy. No entanto, isso não afeta o funcionamento do seu código. O seu código rodou corretamente e a acurácia foi impressa no final. Portanto, você pode ignorar esse aviso por enquanto.

# Carregue o conjunto de dados. Se houver um conjunto de dados atualizado, carregue o atualizado.
customer_ajustado = customer_ajustado


# Separa os atributos das classes.
X = customer_ajustado.drop('label', axis=1)
y = customer_ajustado['label']

# Sem normalizar o conjunto de dados dividido o conjunto de dados em treino e teste.
# Divide o conjunto de dados em treino e teste (70% treino, 30% teste).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Instancia o classificador KNN com k=5 (valor padrão).
knn = KNeighborsClassifier()

# Treina o classificador com o conjunto de treino.
knn.fit(X_train, y_train)

# Implementa o KNN exibindo a acurácia nos dados de teste mantendo a parametrização padrão.
# Faz a predição no conjunto de teste.
y_pred = knn.predict(X_test)

# Calcula a acurácia da predição.
accuracy = accuracy_score(y_test, y_pred)

# Imprime a acurácia.
print('Acurácia sem ser normalizada:', accuracy)
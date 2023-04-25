from src.utils import load_diabetes_clean_dataset
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

diabetes_df = load_diabetes_clean_dataset()
print(diabetes_df.head())
x = diabetes_df.drop("glucose", axis=1).values #vai tirar somente a ccluna da glicose
y = diabetes_df["glucose"].values #vai receeber essa coluna
print("tipo do valor de x:", type(x), "tipo do valor de y:", type(y))
X_bmi = x[:, 3] #vai receber todas as linhas do quarto elemento
print(y.shape, X_bmi.shape) #mostrar o tamanho do array
X_bmi = X_bmi.reshape(-1, 1) #o modelo n pode ser unidimensional, por isso vai ser usado a funçao "reshape" para forcar  o data set a ser bidimensional( no caso com pelo menos uma coluna )
print(X_bmi.shape)

plt.scatter(X_bmi, y) # mostrar os pontinhos no gráfico
plt.ylabel("GLICOSE") # só mostra o nome na vertical que equivale ao "y"
plt.xlabel("BMI") #só mostra o nome na vertical que equivale ao "X"
#plt.show() # mostra o grafico

reg = LinearRegression() # inicializa o modelo
reg.fit(X_bmi, y) #  sempre usamos para ajustar o modelo aos dados que estamos passando
predição = reg.predict(X_bmi) # dado um modelo treinado, preveja o rótulo de um novo conjunto de dados. Este método aceita um argumento, o novo dado X_new(por exemplo model.predict(X_new)), e retorna o rótulo aprendido para cada objeto no array
plt.scatter(X_bmi, y) # mostrar os pontinhos no gráfico
plt.plot(X_bmi, predição)# mostra a linha, no caso a linha de predição
plt.ylabel(" glicose ")# só mostra o nome na vertical que equivale ao "y"
plt.xlabel(" bmi ")#só mostra o nome na vertical que equivale ao "X"
plt.show() #mostra o grafico.

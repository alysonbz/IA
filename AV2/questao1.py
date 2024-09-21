#Verifique qual atributo será o alvo para regressão no seu dataset
#e faça uma análise de qual atributo é mais relevante para realizar a regressão do alvo escolhido.
#Lembre de comprovar via gráfico.
#Obs: Registrar na seção de resultados a análise realizada e discutir sobre o resultado encontrado.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
"""import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.model_selection import train_test_split"""


clean_dataset = pd.read_csv('./dataset/Clean_Dataset.csv')
clean_dataset = clean_dataset.drop(clean_dataset.columns[0], axis=1)

#y = clean_dataset["stops"].values
clean_dataset["airline"] = clean_dataset["airline"].replace("SpiceJet", 0)
clean_dataset["airline"] = clean_dataset["airline"].replace("AirAsia", 1)
clean_dataset["airline"] = clean_dataset["airline"].replace("Vistara", 2)
clean_dataset["airline"] = clean_dataset["airline"].replace("GO_FIRST", 3)
clean_dataset["airline"] = clean_dataset["airline"].replace("Indigo", 4)
clean_dataset["airline"] = clean_dataset["airline"].replace("Air_India", 5)

clean_dataset["departure_time"] = clean_dataset["departure_time"].replace("Evening", 0)
clean_dataset["departure_time"] = clean_dataset["departure_time"].replace("Early_Morning", 1)
clean_dataset["departure_time"] = clean_dataset["departure_time"].replace("Morning", 2)
clean_dataset["departure_time"] = clean_dataset["departure_time"].replace("Afternoon", 3)
clean_dataset["departure_time"] = clean_dataset["departure_time"].replace("Night", 4)


clean_dataset["stops"] = clean_dataset["stops"].replace("zero", 0)
clean_dataset["stops"] = clean_dataset["stops"].replace("one", 1)
clean_dataset["stops"] = clean_dataset["stops"].replace("two_or_more", 2)

clean_dataset["class"] = clean_dataset["class"].replace("Economy", 0)
clean_dataset["class"] = clean_dataset["class"].replace("Business", 1)
print(clean_dataset["class"])

new_clean_dataset = clean_dataset.select_dtypes(include=[float, int])
print(new_clean_dataset)
correlation_matrix = new_clean_dataset.corr()
corr_target = correlation_matrix["price"].sort_values(ascending=False).drop("price")
print(corr_target)


# Exemplo: se 'Duration' for o mais relevante
plt.figure(figsize=(10, 8))
#sns.scatterplot(x='Duration', y='Price', data=df)
sns.heatmap(corr_target.to_frame(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title('Correlação dos atributos com o alvo')
plt.show()


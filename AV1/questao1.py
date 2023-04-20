#importe as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import questao2 as quest2

## Carregue o dataset definido para você
dados_drogas = pd.read_csv(r'C:\Paulo Henrique\IA\AV1\dataset\drug200.csv')


# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(dados_drogas.isnull().sum())
print(dados_drogas.isna().sum())
#Como não há valores vazios, nem Nan, não há necessidade de atuzlizar um Dataframe

# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
Drugs_and_Features = pd.DataFrame(dados_drogas[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug']])

#Print o dataframe final e mostre a distribuição de classes que você deve classificar
print(Drugs_and_Features)
distribuição_de_classes = Drugs_and_Features['Drug'].value_counts()

#Resolvi plotar um gráfico, para ficar mais nítido a diferença de distribuição da classe Drug.
distribuição_de_classes.plot(kind='bar')
plt.xlabel('Classe')
plt.ylabel('Quantidade')
plt.title('Distribuição das Classes')
plt.show()

#observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário
le_sex = LabelEncoder()
le_BP = LabelEncoder()
le_cholesterol = LabelEncoder()

# Aplicar a conversão para valores numéricos nas colunas relevantes
Drugs_and_Features['Sex'] = le_sex.fit_transform(Drugs_and_Features['Sex'])
Drugs_and_Features['BP'] = le_BP.fit_transform(Drugs_and_Features['BP'])
Drugs_and_Features['Cholesterol'] = le_cholesterol.fit_transform(Drugs_and_Features['Cholesterol'])
print(Drugs_and_Features)

mapeamento = {'DrugY': 0, 'drugC': 1, 'drugX': 2, 'drugA': 3, 'drugB': 4}
Drugs_and_Features['Drug'] = Drugs_and_Features['Drug'].map(mapeamento)

print(Drugs_and_Features.head())



#Salve o dataset atualizado se houver modificações.
Drugs_and_Features.to_csv("Drugs_and_Features", index=False)
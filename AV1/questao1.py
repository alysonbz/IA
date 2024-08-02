# Importe as bibliotecas necessárias.
from src.utils import load_heart_dataset


# Carregue o dataset definido para você.
heart = load_heart_dataset()


# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print(heart.isna().sum())

new_heart = heart.dropna(axis=0)
print()

print(new_heart.isna().sum())
print()


# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
print("\n--- COLUNA RELEVANTE OUTPUT ---\n", new_heart['output'])
print()

# Print o dataframe final e mostre a distribuição de classes que você deve classificar.
print("DATAFRAME - HEART:")
print(new_heart.info())
print()

# Calcular a distribuição da coluna 'output'
print("CALCULANDO A DISTRIBUIÇÃO DAS CLASSES:")
class_distribution = new_heart['output'].value_counts()
print(class_distribution, '\n', '\n')

# Observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário.
print("TODAS AS COLUNAS JÁ ESTÃO COM ATRIBUTOS NUMÉRICOS:")
print(new_heart.info())
print()

# Salve o dataset atualizado se houver modificações.
new_heart.to_csv('dataset/new_heart_data.csv', index=False)
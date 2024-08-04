# Importe as bibliotecas necessárias.
from src.utils import load_customer_dataset
import matplotlib.pyplot as plt
import seaborn as sns


# Carregue o dataset definido para você.
customer = load_customer_dataset()
print(customer.head())
print()

# Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.
print("ANALISANDO VALORES NULOS.")
print(customer.isna().sum())
print()

customer = customer.dropna(axis=0)
print("VALORES NULOS DELETADOS.")
print()

print("SEM OS VALORES NULOS NA COLUNA 'fea_2':")
print(customer.isna().sum())
print()


# Verifique quais colunas são as mais relevantes e crie um novo dataframe.
print("\n--- TODAS AS COLUNAS ---\n", customer.columns)

new_customer = customer.drop("id", axis=1)

print("\n--- REMOÇÃO DA COLUNA 'id' / APENAS COLUNAS RELEVANTES---\n", new_customer)
print()


# Print o dataframe final e mostre a distribuição de classes que você deve classificar.
print("DATAFRAME - CUSTOMER:")
print(new_customer)
print()

print("CALCULANDO A DISTRIBUIÇÃO DA CLASSE 'label':")
class_distribution = new_customer
print(class_distribution['label'].value_counts(), '\n','\n')
print()

#print("VISUALIZANDO A DISTRIBUIÇÃO EM UM GRÁFICO:")
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=new_customer, palette='viridis')
plt.title('Distribuição das Classes')
plt.xlabel('Classe')
plt.ylabel('Contagem')
plt.show()

print()

# Observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário.
print("TODAS AS COLUNAS JÁ ESTÃO COM ATRIBUTOS NUMÉRICOS:")
print(new_customer.info())
print()

# Salve o dataset atualizado se houver modificações.
new_customer.to_csv('dataset/new_customer_data.csv', index=False)

#importe as bibliotecas necessárias
from src.utils import load_cancer_dataset
import colorama
## Carregue o dataset definido para você
cancer = load_cancer_dataset()
colorama.init()

#Primeiro vou ter uma noção do dataset através dos comandos abaixo
print(cancer)
print("---"*20)



print ("1.1) Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.")
print(cancer.isnull().sum())
#Conclusão -> existem uma coluna vazia de nome "unnamed: 32"

#Criar um novo dataframe sem a coluna "unnamed:32"
#print(cancer.dropna())
cancer_cleaned_isnull = cancer.drop(columns=['Unnamed: 32'])
print("Dataframe sem a coluna Unnamed: 32: ")
#print(cancer_cleaned_isnull)


print('1.2) Verifique quais colunas são as mais relevantes e crie um novo dataframe.')

print("já foi retirada a coluna Unnamed na seção anterior. Mas além disso, foi retirada a coluna ID pelo fato de não apresentar elementos válidos já que ela identifica outro elemento ")
cancer_cleaned_relevant = cancer_cleaned_isnull.drop(columns=['id'])


print('1.3) Print o dataframe final e mostre a distribuição de classes que você deve classificar')
print("A classe escolhida será diagnosis, assim sendo o y. E todas as outras, caractéristicas(x)")
print(cancer_cleaned_relevant)


print('1.4) Observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário')
print("As classe referentes á classificação Maligna(m) e Benigna(b) foram renomeados para atributos númericos")
cancer_cleaned_relevant['diagnosis'] = cancer_cleaned_relevant['diagnosis'].map({'M':1, 'B':0})
print(cancer_cleaned_relevant['diagnosis'].value_counts())


print('Salve o dataset atualizado se houver modificações.')
cancer_cleaned_relevant.to_csv('dataset/Cancer_Data_Cleaned.csv', index=False)

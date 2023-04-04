# Atividades de Sala
> Orientações para execução das atividades de sala.

Esse documento exibe as descrições das questões a serem resolvidas em sala

##  Unidade 2

### Questão 1

```1_standardization.py```

#### Classificar sem normalizar

Nesta questão você vai realizar uma classificação com KNN sem executar normalizaçao dos atributos.

#### Instruções:

1)  Utilizando a função `` train_test_split `` Divida o dataset em dados para teste e dados para treino. Faça uma amostragem estratificada com base na proporção de label de ``y``.
   
2)  Execute o treinamento do classifcador knn. Este já foi iniccalizado para você com a linha  `` knn = KNeighborsClassifier() ``. Agora você deve chamar a função ``fit`` e colocar como argumentos ``X_train`` e ``y_train``. 

3)  Print a quantidade de elementos que estão faltando na coluna `` locality``.
    
4)  Execute a função ``score`` de ``knn`` para medir a acurácia do classidicador. 

### Questão 2

```2_data_types.py```

#### Instruções 

1) Print os 5 primeiros elementos da coluna ``hits`` do dataset ``volunteer``.
2) Print as caracteristicas da coluna``hits``.
3) Converta a coluna ``hits`` de ``volunteer`` para o tipo ``int32``.
4) Print as caracteristicas da coluna``hits`` novamente.


### Questão 3

```3_training_and_test_sets.py```

#### Treinando um dataset

Neste exercicio você irá aprender a separar o dataset em um conjunto para treino e teste.

#### Instruções

1) importe a função ``train_test_split``
2) Exclua as colunas ``Latitude`` e ``Longitude`` de volunteer com a função ``drop``.
3) Exclua as linhas com valores ``null`` da coluna ``category_desc`` de ``volunteer_new``
4) Mostre o balanceamento das classes em ``category_desc`` utilizando a função ``value_counts``
5) Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
6) Crie um dataframe de labels com a coluna ``category_desc``
7) Utiliza a a amostragem stratificada para separar o dataset em treino e teste
8) Mostre o balanceamento das classes em ``category_desc`` novamente

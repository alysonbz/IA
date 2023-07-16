# AVALIAÇÃO FINAL
> Orientações para execução da prova.

Esse documento exibe as descrições das questões e a relação dos datasets que devem ser utiizados 
pelos alunos e alunas.

O modelo de documento seguinte mostra como você deve registrar por escrito o desenvolvimento. 
https://docs.google.com/document/d/1hIwPx9W-k3LnXRJrkWYTsbrtx4NfP88_/edit?usp=sharing&ouid=118351454454462119994&rtpof=true&sd=true

##  Dataset

https://www.kaggle.com/datasets/utkarshsaxenadn/flower-classification-5-classes-roselilyetc

### Questão 1

```questao1.py```

Neste primeiro exercício você deve realizar manipulação em um dataset com a biblioteca pandas e realizar o pré-processamento deste.

#### Instruções:

1) Importe as bibliotecas necessárias.
   
2) Verifique se existem celulas vazias ou Nan. Se existir, excluir e criar um novo dataframe.

3) Verifique quais colunas são as mais relevantes e crie um novo dataframe com somente as colunas necesárias. 
    
4) Print o dataframe final e mostre a distribuição de classes que você deve classificar

5) Observe se a coluna de classes precisa ser renomeada para atributos numéricos, realize a conversão, se necessário

6) Salve o dataset atualizado se houver modificações. Faça uma renomeação para ``nome_do_dataset_ajustado.csv``

### Questão 2

```questao2.py```

neste segundo exercício você deve realizar uma classificação utilizando KNN.

#### Instruções 

1) Importe as bibliotecas necessárias.
2) Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
3) Sem normalizar o conjunto de dados divida o dataset em treino e teste.
4) Implemente o Knn exbindo sua acurácia nos dados de teste e mantenha sua parametrização default.


### Questão 3

```questao3.py```

Neste exercicio você deve verificar se a normalização interfere nos resultados de sua classificação.

#### Instruções

1) Importe as bibliotecas necessárias.
2) Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
3) Normalize o conjunto de dados com normalização logarítmica  e verifique a acurácia do knn.
4) Normalize o conjunto de dados com normalização de media zero e variância unitária e e verifique a acurácia do knn.
5) Print as duas acuracias lado a lado para comparar. 


### Questão 4

```questao4.py```

Neste exercicio vc deve buscar saber a melhor parametrização do knn.

#### Instruções

1) Importe as bibliotecas necessárias.
2) Carregue o dataset. Se houver o dataset atualizado, carregue o atualizado.
3) Normalize com a melhor normalização o conjunto de dados se houver melhoria.
4) Plote o gráfico com o a indicação do melhor k.


### Questão 5

```questao5.py```

Neste exercicio vc deve realizar uma redução de dimensão com PCA. Com base na análise de variância, reduza a dimensão
e compare o resutado do KNN antes e depois da redução de dimensão.


### Observações para o Relatório

Discutir **organizadamente** na sessão de resultados os números obtidos de cada questão.
Ao concluir o relatório, compartilhar com **alysonbnr@ufc.br**
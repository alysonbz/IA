# AVALIAÇÃO 1
> Orientações para execução da prova.

Esse documento exibe as descrições das questões e a relação dos datasets que devem ser utiizados 
pelos alunos e alunas.

O modelo de documento seguinte mostra como você deve registrar por escrito o desenvolvimento. 
https://docs.google.com/document/d/1hIwPx9W-k3LnXRJrkWYTsbrtx4NfP88_/edit?usp=sharing&ouid=118351454454462119994&rtpof=true&sd=true

##  Aluno - Dataset

ANA LIVIA SOUSA DAVI TAVEIRA : https://www.kaggle.com/datasets/erdemtaha/cancer-data

[Cancer_Data.csv](dataset%2FCancer_Data.csv)

CARLOS EDUARDO TELES ALENCAR: https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset

[Hotel Reservations.csv](dataset%2FHotel%20Reservations.csv)

DAVI GONCALVES RAMOS DE MESQUITA: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset

[heart.csv](dataset%2Fheart.csv)

EMILY CAMELO MENDONCA: https://www.kaggle.com/datasets/elakiricoder/gender-classification-dataset

[gender_classification_v7.csv](dataset%2Fgender_classification_v7.csv)

ERICK RAMOS COUTINHO: https://www.kaggle.com/datasets/jillanisofttech/brain-tumor

[data_brain_tumor.csv](dataset%2Fdata_brain_tumor.csv)


ERYKA CARVALHO DA SILVA:https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

[airline_satisfaction.csv](dataset%2Fairline_satisfaction.csv)


GIOVANNA DIAS CASTRO DE OLIVEIRA:https://www.kaggle.com/datasets/nextbigwhat/dataset-1?select=dataset_1.csv

[dataset__binary.csv](dataset%2Fdataset__binary.csv)

LARISSA VITORIA VASCONCELOS SOUSA: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

[healthcare-dataset-stroke-data.csv](dataset%2Fhealthcare-dataset-stroke-data.csv)

LUCIANA SOUSA MARTINS: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

[heart_desease.csv](dataset%2Fheart_desease.csv)

LUIS SAVIO GOMES ROSA: https://www.kaggle.com/datasets/rtatman/chocolate-bar-ratings

[flavors_of_cacao.csv](dataset%2Fflavors_of_cacao.csv)

MARIA BIANCA SOUSA COSTA: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

[diabetes_012_health_indicators_BRFSS2015.csv](dataset%2Fdiabetes_012_health_indicators_BRFSS2015.csv)

MAVERICK ALEKYNE DE SOUSA RIBEIRO:https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

[breast-cancer.csv](dataset%2Fbreast-cancer.csv)

PAULO HENRIQUE SANTOS MARQUES: https://www.kaggle.com/datasets/prathamtripathi/drug-classification

[drug200.csv](dataset%2Fdrug200.csv)


RUAN RODRIGUES SOUSA: https://www.kaggle.com/datasets/zaraavagyan/weathercsv

[weather.csv](dataset%2Fweather.csv)

SHELDA DE SOUZA RAMOS: https://www.kaggle.com/datasets/whenamancodes/predict-diabities

[diabetes.csv](dataset%2Fdiabetes.csv)

THAIS ANDRADE CASTRO: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

[star_classification.csv](dataset%2Fstar_classification.csv)

VICTOR MATHEUS ARAUJO OLIVEIRA: https://www.kaggle.com/datasets/praveengovi/credit-risk-classification-dataset?select=customer_data.csv

[customer_data.csv](dataset%2Fcustomer_data.csv)

WILLIAN KELVIN BORGES DA COSTA: https://www.kaggle.com/datasets/mssmartypants/water-quality

[waterQuality1.csv](dataset%2FwaterQuality1.csv)


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


### Observações para o Relatório

Discutir **organizadamente** na sessão de resultados os números obtidos de cada questão.
Ao concluir o relatório, compartilhar com **alysonbnr@ufc.br**
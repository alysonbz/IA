# AVALIAÇÃO 2 -prazo para envio do relatório e código : 31/05/2023
> Orientações para execução da prova.

Esse documento exibe as descrições das questões e a relação dos datasets que devem ser utiizados 
pelos alunos e alunas.

O modelo de documento seguinte mostra como você deve registrar por escrito o desenvolvimento. 
https://docs.google.com/document/d/1hIwPx9W-k3LnXRJrkWYTsbrtx4NfP88_/edit?usp=sharing&ouid=118351454454462119994&rtpof=true&sd=true

##  Aluno - Dataset

ANA LIVIA SOUSA DAVI TAVEIRA :https://www.kaggle.com/datasets/anubhavgoyal10/laptop-prices-dataset

CARLOS EDUARDO TELES ALENCAR: https://www.kaggle.com/datasets/kapturovalexander/hp-lenovo-acer-asus-samsung-companies-share-prices

DAVI GONCALVES RAMOS DE MESQUITA:https://www.kaggle.com/datasets/kapturovalexander/ferrari-and-tesla-share-prices-2015-2023

EMILY CAMELO MENDONCA: https://www.kaggle.com/datasets/rkiattisak/smart-watch-prices

ERICK RAMOS COUTINHO: https://www.kaggle.com/datasets/thaweewatboy/thailand-carbon-emission-statistics

ERYKA CARVALHO DA SILVA:https://www.kaggle.com/datasets/kapturovalexander/lg-and-samsung-share-prices-2002-2023

GIOVANNA DIAS CASTRO DE OLIVEIRA: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

LARISSA VITORIA VASCONCELOS SOUSA: https://www.kaggle.com/datasets/harshghadiya/car-price-prediction

LUCIANA SOUSA MARTINS: https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction

LUIS SAVIO GOMES ROSA: https://www.kaggle.com/datasets/kapturovalexander/activision-nintendo-ubisoft-ea-stock-prices

MARIA BIANCA SOUSA COSTA: https://www.kaggle.com/datasets/yasserh/student-marks-dataset

MAVERICK ALEKYNE DE SOUSA RIBEIRO: https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge

PAULO HENRIQUE SANTOS MARQUES: https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles

RUAN RODRIGUES SOUSA: https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset

SHELDA DE SOUZA RAMOS: https://www.kaggle.com/datasets/equilibriumm/sleep-efficiency

THAIS ANDRADE CASTRO: https://www.kaggle.com/datasets/prasertk/average-screen-time-and-usage-by-country

VICTOR MATHEUS ARAUJO OLIVEIRA: https://www.kaggle.com/code/shubhammeshram579/forestfires-prediction

WILLIAN KELVIN BORGES DA COSTA: https://www.kaggle.com/datasets/rkiattisak/mobile-phone-price

[waterQuality1.csv](dataset%2FwaterQuality1.csv)


### Questão 1

```questao1.py```

Verifique qual atributo será o alvo para regressão no seu dataset
e faça uma análise de qual atributo é mais relevante para realizar a regressão do alvo escolhido.
Lembre de comprovar via gráfico.
Obs: Registrar na seção de resultados a análise realizada e discutir sobre o resultado encontrado.


### Questão 2

```questao2.py```

Utilizando o atributo mais relevante calculado na questão 1, implemente uma regressão linear utilizando somente este atributo mais
relevante, para predição do atributo alvo determinado na questão 1 também. Mostre o gráfico da reta de regressão  em conjunto com a nuvem 
de atributo. 
Determine também os valores: 
RSS, MSE, RMSE e R_squared para esta regressão baseada somente no atributo mais relevante.
Obs: Registrar na seção de resultados a análise realizada e discutir sobre os resultados encontrados.

### Questão 3

```questao3.py```

Remova os atributos que não são relevantes para o processo de regressão e realize um gridsearch cross-validation para verificar 
qual a melhor parametrização para os regressores de Lasso e Ridge. Print as melhores configurações de cada um mostre também os melhores scores.
Obs: Registrar na seção de resultados a análise realizada e discutir sobre os resultados encontrados.

### Questão 4

```questao4.py```

Utilizando kfold e cross-validation faça uma regressão linear utilizando os mesmos atributos definidos na questão 3.
Obs: Com os resultados obtidos na questão 3 e da questão 4 faça uma comparação entre os desempenhos. Escolha o regressor adequado
e informe o motivo da escolha. Discuta sobre as limitações e acertos encontrados.


### Observações para o Relatório

Discutir **organizadamente** na sessão de resultados os números obtidos de cada questão.
Ao concluir o relatório, compartilhar com **alysonbnr@ufc.br**
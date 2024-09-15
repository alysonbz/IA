# AVALIAÇÃO 2 -prazo para envio do relatório e código : 20/09/20
> Orientações para execução da prova.

Esse documento exibe as descrições das questões e a relação dos datasets que devem ser utiizados 
pelos alunos e alunas.

O modelo de documento seguinte mostra como você deve registrar por escrito o desenvolvimento. 
https://docs.google.com/document/d/1hIwPx9W-k3LnXRJrkWYTsbrtx4NfP88_/edit?usp=sharing&ouid=118351454454462119994&rtpof=true&sd=true

##  Aluno - Dataset

ALLAN MICHEL ROCHA DOS SANTOS :https://www.kaggle.com/datasets/anubhavgoyal10/laptop-prices-dataset

ANTONIO FILIPE SOUSA SILVA : https://www.kaggle.com/datasets/kapturovalexander/hp-lenovo-acer-asus-samsung-companies-share-prices

FRANCISCO SAMUEL SALES PINHEIRO PINTO :https://www.kaggle.com/datasets/kapturovalexander/ferrari-and-tesla-share-prices-2015-2023

GUILHERME PINHEIRO SERAFIM: https://www.kaggle.com/datasets/rkiattisak/smart-watch-prices

JOAO LUIS FEITOSA LEITE: https://www.kaggle.com/datasets/thaweewatboy/thailand-carbon-emission-statistics

JOAO PAULO ROCHA MATOS:https://www.kaggle.com/datasets/kapturovalexander/lg-and-samsung-share-prices-2002-2023

JURANDIR CAVALCANTI GOMES NETO: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final

PEDRO JONNATHAN MATOS DE SOUSA: https://www.kaggle.com/datasets/harshghadiya/car-price-prediction

VITORIA NASCIMENTO DE PAULA: https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction

WANDERSON WENDEL DE SOUSA LOPES: https://www.kaggle.com/datasets/kapturovalexander/activision-nintendo-ubisoft-ea-stock-prices


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
# AVALIAÇÃO 3 - Prazo para envio do relatório e código : 04/07/2023
> Orientações para execução da prova.

Esse documento exibe as descrições das questões e a relação dos datasets que devem ser utiizados 
pelos alunos e alunas.

O modelo de documento seguinte mostra como você deve registrar por escrito o desenvolvimento. 
https://docs.google.com/document/d/1hIwPx9W-k3LnXRJrkWYTsbrtx4NfP88_/edit?usp=sharing&ouid=118351454454462119994&rtpof=true&sd=true

##  Aluno - Dataset

ANA LIVIA SOUSA DAVI TAVEIRA :https://www.kaggle.com/datasets/uciml/mushroom-classification

CARLOS EDUARDO TELES ALENCAR: https://www.kaggle.com/datasets/uciml/glass

DAVI GONCALVES RAMOS DE MESQUITA:https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification

EMILY CAMELO MENDONCA: https://www.kaggle.com/datasets/christianlillelund/csgo-round-winner-classification

ERICK RAMOS COUTINHO: https://www.kaggle.com/datasets/olcaybolat1/dermatology-dataset-classification

ERYKA CARVALHO DA SILVA: https://www.kaggle.com/datasets/gyejr95/league-of-legends-challenger-ranked-games2020

GIOVANNA DIAS CASTRO DE OLIVEIRA: https://www.kaggle.com/datasets/sadmansakib7/ecg-arrhythmia-classification-dataset

LARISSA VITORIA VASCONCELOS SOUSA: https://www.kaggle.com/datasets/shebrahimi/financial-distress

LUCIANA SOUSA MARTINS: https://www.kaggle.com/datasets/kevinarvai/clinvar-conflicting

LUIS SAVIO GOMES ROSA: https://www.kaggle.com/datasets/sudhanshu2198/oil-spill-detection

MARIA BIANCA SOUSA COSTA: https://www.kaggle.com/datasets/robikscube/eye-state-classification-eeg-dataset?select=EEG_Eye_State_Classification.csv

MAVERICK ALEKYNE DE SOUSA RIBEIRO: https://www.kaggle.com/datasets/gauravduttakiit/dry-bean-classification

PAULO HENRIQUE SANTOS MARQUES: https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

RUAN RODRIGUES SOUSA: https://www.kaggle.com/datasets/shayanfazeli/heartbeat

SHELDA DE SOUZA RAMOS: https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking

THAIS ANDRADE CASTRO: https://www.kaggle.com/datasets/abrahamanderson/cancer-classification

VICTOR MATHEUS ARAUJO OLIVEIRA: https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset

WILLIAN KELVIN BORGES DA COSTA: https://www.kaggle.com/datasets/subhajournal/trojan-detection




### Questão 1

```questao1.py```

Faça uma análise do dataset utilizando dendograma. Verifique as possibilidades 
de clusterização e aplique o k-medias. Observe os resultados e descreva sua iterpretação
no relatório.
Dica: Observe se há necessidade de normalização dos dados nas colunas ou nas linhas.


### Questão 2

```questao2.py```
Reduza o dataset T-SNE e com PCA para duas dimensões. Plote o gráfico do atributos que as duas técnicas geraram.
De forma subjetiva e visual, qual dos gráficos você avredita que vai possuir um melhor
desempenho em um processo de classificação utilizando os dois atribuitos ?

### Questão 3

```questao3.py```

Utilizando os dados da questão 2, aplique algum método de classificação e gere números
que quantificam o desempenho deste. Compare os números classificando o dataset reduzido pelo PCA e pelo T-SNE.


### Questão 4

```questao4.py```

Utilizando análise de variância do PCA. Reduza a dimensão para realizar uma classificação utilizando somente as colunas de maior variância.
Aplique o mesmo método de classificação testado na questão 3. Gere os mesmos números que analisam o desempenho do classificador e verifique se houve melhoria no resultado.


### Questão 5

```questao5.py```

Você descobriu qual a melhor forma de pré-processar os dados. Assim, utilizando a metodologia que proporcionou o melhor acerto do classficador faça agora uma comparação 
entre classicadores para que você também possa descobrir qual classificador mais adequado. Utilize outra técnica de classificação com os mesmo dados, gere os numeros que quantificam o 
desempenho e faça uma comparação entre estes.
Conclua o relatótório  com auxílio de um fluxogragrama mostrando qual a metodologia completa para classificação dos dados do seu dataset.


### Observações para o Relatório

Discutir **organizadamente** na sessão de resultados os números obtidos de cada questão.
Ao concluir o relatório, compartilhar com **alysonbnr@ufc.br**
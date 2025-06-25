# Dianóstico de Parkinson baseado em Redes Cerebrais Funcionais

Este repositório faz parte da entrega do trabalho da disciplina SCC0270 - Redes Neurais e Aprendizado Profundo (2025)

    Francisco Luiz Maian do Nascimento - 14570890
    Gabriel da Costa Merlin - 12544420
    João Pedro Soares de Azevedo Calixto - 13732011
    Théo Bruno Frey Riffel - 12547812
    Vítor Amorim Fróis - 12543440

## Projeto

Este repositório contém um projeto para o diagnóstico da doença de Parkinson (DP) usando aprendizado profundo.

A doença de Parkinson é uma doença neurológica que afeta os movimentos da pessoa. Nesse sentido, a aplicação de Redes Neurais Profundas se destaca como uma ferramenta para identificação precoce da doença. A doença de Parkinson afeta a região cerebral, assim buscamos treinar uma Rede Neural que identifica a doença em dados cerebrais dos novos pacientes.

A ressonância magnética funcional (fMRI) mede o sinal dependente do nível de oxigênio no sangue (BOLD). Os disparos neurais produzem alterações detectáveis no fMRI. Assim, o sinal serve como um indicador indireto da atividade neural, baseado no fluxo sanguíneo e da oxigenação.

Nesse projeto, utilizamos dados de fMRI recolhidos com pacientes em repouso. Os dados de rs-fMRI (resting state functional Magnetic Ressonance Imaging) foram obtidos da [base de dados PPMI](https://www.ppmi-info.org/PPMI) e pré-processados com a ferramenta [CONN toolbox](https://web.conn-toolbox.org/) para remover ruídos de movimento físico dos pacientes durante o exame, ruídos provenientes da máquina e de efeitos adversos do campo magnético. Para acessar os dados, contate um dos contribuidores do projeto. As amostras para cada grupo são exibidas abaixo.

| Controle 	| Parkinson 	|
|:--------:	|-----------	|
| 66       	| 153       	|

Nos últimos anos, técnicas avançadas de neuroimagem se tornaram fundamentais na busca pela identificação não invasiva da DP. Na área de Conectomas Cerebrais, utilizam-se dados rs-fMRI para entender a organização, funcionalidade e comportamento do cérebro humano. A principal hipótese explorada é de que a conectividade do cérebro é alterada para pacientes que apresentam a doença de Parkinson ([Connectomics: a new paradigm for understanding brain disease](https://pubmed.ncbi.nlm.nih.gov/24726580/), [Human brain networks in health and disease](https://pubmed.ncbi.nlm.nih.gov/19494774/)).

Para tanto, é possível associar os dados de rs-fMRI com cada região do cérebro, através da agregação dos voxels em cada instante de tempo. Assim, obtemos uma série temporal multivariada para cada paciente, onde cada canal corresponde a uma região. Dado a hipótese de conectividade, podemos utilizar métricas de similaridade, como Dynamic Time Warping, Correlação de Pearson e iCOH para obter a matriz $A$ de correlação entre as séries. 

A matriz simétrica $A_{N \times N}$ recebe o nome de **Matriz de Conectividade**, onde cada elemento $a_{ij}$ corresponde ao grau de dependência funcional entre a região $i$ e a região $j$. Isso é, se o sinal BOLD da região $i$ é similar ao sinal BOLD da região $j$, dizemos que essas regiões compartilham funções parecidas no cérebro. 

Assim, cada notebook explora uma abordagem para identificação da doença de Parkinson a partir da Matriz de Conectividade e estão disponíveis no diretório `notebooks`.

* **[1_correlation_matrix.ipynb](notebooks/1_correlation_matrix.ipynb):** é proposta uma baseline para a tarefa. Para isso, uma rede neural foi treinada diretamente sobre a triangular superior da matriz de correlação. Foram exploradas diversas métricas para a construção da matriz. A rede escolhida foi uma MLP simples, com Dropout e uma camada oculta.
* **[2_time_series.ipynb](notebooks/2_time_series.ipynb):** esse notebook explora uma abordagem sem a matriz de conectividade. É esperado um desempenho pior, dado as hipóteses de conectividade. Efetivamente, a MLP treinada diretamente nas séries temporais das Regiões de Interesse não consegue aprender os padrões da doença de Parkinson.
* **[3_gnn.ipynb](notebooks/3_gnn.ipynb):** para aumentar a acurácia na tarefa de classificação, é aplicado um modelo mais robusto para identificação de doenças através da conectividade, inspirado em estudos com dados similares em outras doenças ([ The Combination of a Graph Neural Network Technique and Brain Imaging to Diagnose Neurological Disorders: A Review and Outlook ](https://pubmed.ncbi.nlm.nih.gov/37891830/)). Para tanto, é construído um grafo robusto com Spanning Trees. Os vértices são caracterizados utilizando a correlação e GNNs são aplicadas para classificação. 

## Análise dos métodos

### Notebook 1: 1_correlation_matrix.ipynb (Matriz de Correlação + MLP)
Este notebook aborda o problema de classificação usando uma abordagem baseada em conectividade funcional dinâmica.

1. Metodologia:

Extração de Características: A principal característica extraída das séries temporais de fMRI é a matriz de correlação funcional. O método sliding_window_correlation (correlação com janela deslizante) é utilizado, o que captura a natureza dinâmica da conectividade cerebral ao longo do tempo, em vez de uma correlação estática única.

Modelo: As matrizes de correlação são então "achatadas" (vetorizadas) e usadas como entrada para um MLP (Multi-Layer Perceptron), uma rede neural totalmente conectada padrão.

Tratamento de Dados: O desbalanceamento de classes é tratado usando RandomOverSampler, que replica aleatoriamente amostras da classe minoritária no conjunto de treinamento.

2. Resultados:

Acurácia: 77,27%

F1-Score: 71,75%

Precisão: 82,82%

Recall: 77,27%

### Notebook 2: 2_time_series.ipynb

Este notebook tenta classificar os pacientes usando diretamente os dados de séries temporais, uma abordagem de ponta a ponta ("end-to-end").

1. Metodologia:

Extração de Características: Nenhuma característica complexa é pré-calculada. O modelo utiliza as séries temporais brutas de cada ROI (Região de Interesse) do cérebro como entrada.

Modelo: É utilizada uma FCN (Fully Convolutional Network), que é um tipo de Rede Neural Convolucional (CNN) 1D, adequada para extrair padrões diretamente de dados sequenciais como séries temporais.

Tratamento de Dados: O desbalanceamento de classes no conjunto de treinamento é tratado com a técnica SMOTE (Synthetic Minority Over-sampling Technique), que cria amostras sintéticas da classe minoritária.

2. Resultados:

Acurácia: 56,82%

F1-Score: 58,41%

Precisão: 61,33%

Recall: 56,82%

### Notebook 3: 3_gnn.ipynb

Este é o notebook mais complexo, modelando o cérebro como um grafo e aplicando uma Rede Neural de Grafos (GNN).

1. Metodologia:

Construção do Grafo:

Para cada paciente, uma matriz de correlação estática (pearson) é calculada.

Essa matriz densa é então "podada" usando o algoritmo OMST (Orthogonal Minimal Spanning Tree). O objetivo é criar um "backbone" esparso e eficiente da rede cerebral, mantendo apenas as conexões mais importantes, otimizando um balanço entre custo e eficiência da rede (GCE - Global Cost Efficiency).

Características dos Nós (Node Features): As características de cada nó (ROI) são definidas como o seu "perfil de conexão", que corresponde à linha inteira da matriz de correlação original e densa.

Modelo: Uma GCN (Graph Convolutional Network) é usada para a classificação. O modelo aprende a partir da estrutura do grafo OMST (as conexões) e das características dos nós (os perfis de conexão).

Tratamento de Dados: O desbalanceamento é tratado no DataLoader usando um WeightedRandomSampler.

2. Resultados:

Acurácia: 38,64%

F1-Score: 30,60%

AUC: 59,55% (0.59)

## Resultados

Conclusões Comparativas:

A Melhor Abordagem: O Notebook 1 (MLP com matriz de correlação de janela deslizante) apresentou o melhor desempenho. Isso sugere fortemente que a conectividade funcional dinâmica é a característica mais informativa para esta tarefa de classificação.

Complexidade vs. Desempenho: A abordagem mais complexa e teoricamente avançada (Notebook 3, GNN) teve o pior desempenho. Isso é, mais complexidade não garantiu melhores resultados.


| Característica | Notebook 1 (MLP) | Notebook 2 (FCN) | Notebook 3 (GNN) |
| :--- | :--- | :--- | :--- |
| **Abordagem** | Conectividade Funcional Dinâmica | Série Temporal Bruta | Estrutura de Grafo (Backbone) |
| **Modelo** | MLP | FCN (CNN 1D) | GCN (Rede Neural de Grafo) |
| **Complexidade** | Moderada | Moderada | Alta |
| **Acurácia (Teste)** | **77,27%** | 56,82% | 38,64% |
| **F1-Score (Teste)** | **71,75%** | 58,41% | 30,60% |
| **Desempenho Geral** | **Bom** | Moderado | Fraco |


## Configuração e Instalação

Siga estes passos para configurar e executar o projeto em sua máquina local.

### Baixando o código:
``` bash
git clone https://github.com/TheoRiffel/Parkinson-Diagnosis-Deeplearning.git
cd Parkinson-Diagnosis-Deeplearning
```
### Baixando os dados:
Enviamos um arquivo .zip com os dados pelo e-Disciplinas. Basta baixá-lo, colocá-lo na pasta data/ e extraí-lo.

### Criando o ambiente virtual `parkinson` e instalando dependências:
``` bash
make create-env
make module # cria o pacote parkinson
```

Com a configuração concluída, você pode executar os notebooks no diretório 'notebooks'.

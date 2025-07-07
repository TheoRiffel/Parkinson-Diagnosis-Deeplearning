
# Dianóstico de Parkinson baseado em Redes Cerebrais Funcionais

Este repositório faz parte da entrega do trabalho da disciplina SCC0270 - Redes Neurais e Aprendizado Profundo (2025)

    Francisco Luiz Maian do Nascimento - 14570890
    Gabriel da Costa Merlin - 12544420
    João Pedro Soares de Azevedo Calixto - 13732011
    Théo Bruno Frey Riffel - 12547812
    Vítor Amorim Fróis - 12543440

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

Primeiro, [instalar o Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

Quando não for mais executar o projeto, deletar o ambiente criado:
``` bash
make all
```
Depois, basta selecionar o ambiente `parkinson` dentro dos `kernels do jupyter`

#### Depois de utilizar, executar o comando abaixo para excluir o que foi instalado
``` bash
make clean
```

Com a configuração concluída, você pode executar os notebooks no diretório 'notebooks'.

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

### [1_correlation_matrix.ipynb](notebooks/1_correlation_matrix.ipynb)
É proposta uma baseline para a tarefa. Para isso, uma rede neural foi treinada diretamente sobre a triangular superior da matriz de correlação achatada. O desbalanceamento de classes é abordado utilizando `RandomOverSampler`, que replica aleatoriamente amostras da classe minoritária no conjunto de treinamento. Foram exploradas diversas métricas para a construção da matriz. A rede escolhida foi uma rede Multilayer Perceptron simples, com Dropout e uma camada oculta.

### [2_time_series.ipynb](notebooks/2_time_series.ipynb)
Esse notebook explora uma abordagem sem a matriz de conectividade - ao invés disso, são utilizadas as séries ROI como entrada. É esperado um desempenho pior, dado as hipóteses de conectividade. O desbalanceamento de classes no conjunto de treinamento é tratado com a técnica SMOTE (Synthetic Minority Over-sampling Technique), que cria amostras sintéticas da classe minoritária.
Efetivamente, a Fully Convolutional Network treinada diretamente nas séries temporais das Regiões de Interesse não consegue aprender os padrões da doença de Parkinson.

### [3_gnn.ipynb](notebooks/3_gnn.ipynb)
Para aumentar a acurácia na tarefa de classificação, é aplicado um modelo mais robusto para identificação de doenças através da conectividade, inspirado em estudos com dados similares em outras doenças ([ The Combination of a Graph Neural Network Technique and Brain Imaging to Diagnose Neurological Disorders: A Review and Outlook ](https://pubmed.ncbi.nlm.nih.gov/37891830/)). Para tanto, é construído um grafo esparso e eficiente da rede cerebral, mantendo apenas as conexões mais importantes, em um balanço entre custo e eficiência da rede. Os vértices são caracterizados utilizando a linha correspondente da matriz de correlação. O desbalanceamento é tratado no `DataLoader` usando um `WeightedRandomSampler`. GNNs são aplicadas para classificação. 

## Resultados
### Matriz de Correlação
| Acurácia | F1-Score | Precisão | Recall |
|:--------:|----------|----------|--------|
| 77,27%   | 71,75%   | 82,82%   | 77,27% |

### FCN
| Acurácia | F1-Score | Precisão | Recall |
|:--------:|----------|----------|--------|
| 56,82%   | 58,41%   | 61,33%   | 56,82% |

### GCN
| Acurácia | F1-Score | AUC    |
|:--------:|----------|--------|
| 38,64%   | 30,60%   | 59,55% |

A abordagem proposta no notebook 1 apresentou o melhor desempenho. Isso sugere fortemente que a conectividade funcional dinâmica é a característica mais informativa para esta tarefa de classificação. Já a abordagem mais complexa e teoricamente avançada (Notebook 3) teve o pior desempenho.

| Característica | Notebook 1 (MLP) | Notebook 2 (FCN) | Notebook 3 (GNN) |
| :--- | :--- | :--- | :--- |
| **Abordagem** | Conectividade Funcional Dinâmica | Série Temporal Bruta | Estrutura de Grafo (Backbone) |
| **Modelo** | MLP | FCN (CNN 1D) | GCN (Rede Neural de Grafo) |
| **Complexidade** | Moderada | Moderada | Alta |
| **Acurácia (Teste)** | **77,27%** | 56,82% | 38,64% |
| **F1-Score (Teste)** | **71,75%** | 58,41% | 30,60% |
| **Desempenho Geral** | **Bom** | Moderado | Fraco |

## Conclusão
Todos os modelos sofreram com **overfitting**, evidenciado pela rapidez da acurácia de treinamento para alcançar 100% e pela rapidez da subida do erro de validação. Para lidar com isso, os modelos foram simplificados. Mesmo assim, o problema do overfitting persistiu. Assim, o modelo mais simples atingiu o melhor resultado. Futuramente, os notebooks serão adaptados com modelos mais robustos que capturem insights nos dados sem aumentar a complexidade de forma excessiva.

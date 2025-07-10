# Dianóstico de Parkinson baseado em Redes Cerebrais Funcionais

Este repositório faz parte da entrega do trabalho da disciplina SCC0270 - Redes Neurais e Aprendizado Profundo (2025) e foi desenvolvido pelos alunos:

    Francisco Luiz Maian do Nascimento - 14570890
    Gabriel da Costa Merlin - 12544420
    João Pedro Soares de Azevedo Calixto - 13732011
    Théo Bruno Frey Riffel - 12547812
    Vítor Amorim Fróis - 12543440

## Configuração e Instalação

### Baixando o código:
``` bash
git clone https://github.com/TheoRiffel/Parkinson-Diagnosis-Deeplearning.git
cd Parkinson-Diagnosis-Deeplearning
```

### Baixando os dados:
Enviamos um arquivo .zip com os dados pelo e-Disciplinas. Basta baixá-lo, colocá-lo na pasta data/ e extraí-lo.

### Criando o ambiente virtual `parkinson` e instalando dependências:

1. [Instalar o Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

2. Criar o ambiente 'parkinson'
``` bash
make all
```

3. Selecionar o ambiente `parkinson` dentro dos `kernels do jupyter`


Com a configuração concluída, você pode executar os notebooks no diretório 'notebooks'.


- Depois de utilizar, executar o comando abaixo para excluir o que foi instalado    
``` bash
make clean
```

## Sobre o projeto

Este repositório contém um projeto para o diagnóstico multiclasse da doença de Parkinson (DP) a partir de dados de ressonância magnética funcional em repouso (rs-fMRI), baseado em três abordagens principais:

1. **Séries Temporais**

   * Extraímos as séries temporais BOLD (Blood-Oxygen-Level-Dependent) de cada Região de Interesse (ROI) para cada paciente.

   * Comparamos duas estratégias de classificação:
     * **Fully Convolutional Network (FCN):** modelos convolucionais 1D treinados diretamente sobre as séries temporais.
     * **Catch22:** conjunto de 22 métricas extraídas das séries, usadas como características em classificadores simples.

   * Realizamos experimentos de classificação:
     * Binária (Controles vs. Parkinson avançado)
     * Multiclasse (Controles, Prodomal, Parkinson)
     * Classificação em duas etapas (Controle vs. Doente; Prodomal vs. Parkinson)

2. **Matriz de Conectividade**

   * Construímos a Matriz de Conectividade funcional correlacionando as séries BOLD das ROIs usando métricas de similaridade:

     * Pearson, Spearman, Sliding Window, Dynamic Time Warping (DTW) e Imaginary Coherence (iCOH).

  * Comparamos duas estratégias de classificação:

    * Aplicamos redes Multilayer Perceptron (MLP) diretamente sobre a metade superior da matriz (vetorizada), tratando o desbalanceamento com `RandomOverSampler` ou SMOTE.

    * Além de utilizar apenas a matriz de conectividade das séries BOLD das ROIs, também extraímos características das séries BOLD usando o método Catch22. Com essas características, construímos uma nova matriz de conectividade, cuja metade superior, vetorizada, foi usada como entrada para uma MLP simples.

   * Aqui, também realizamos experimentos de classificação:

     * Binária (Controles vs. Parkinson avançado)
     * Multiclasse (Controles, Prodomal, Parkinson)
     * Classificação em duas etapas (Controle vs. Doente; Prodomal vs. Parkinson)

3. **Rede Cerebral (Grafos)**

   * A partir da matriz de conectividade, geramos grafos representando o cérebro de cada paciente:

     * **OMST (Orthogonal Minimum Spanning Tree):** gera uma árvore de custo mínimo.
     * **Thresholding:** conecta regiões cujos coeficientes de correlação excedem um limiar.
   * Extraímos características dos grafos (autovalores da Matriz Laplaciana) e classificamos com Graph Neural Networks (GNN), usando `WeightedRandomSampler` no `DataLoader` para balanceamento.

4. **Multimodalidade**

Para atender à demanda por multimodalidade no trabalho, optamos por realizar um stacking dos melhores modelos obtidos. A partir dos experimentos realizados, observamos que os melhores resultados para o problema de classificação em três classes foram alcançados utilizando, em conjunto, as matrizes de correlação e as séries temporais brutas em uma única etapa de treinamento (notebooks 1.1_3classes e 2.1_3classes).

Assim, combinamos as previsões desses dois modelos por meio de um modelo simples de aprendizado de máquina, escolhemos a regressão logística pela sua simplicidade e bom desempenho.

Os resultados indicam que o modelo multimodal não apresentou ganhos significativos em relação ao modelo baseado apenas nas matrizes de correlação (notebook 1.1_3classes). Em várias execuções, o desempenho do stacking foi frequentemente inferior ou, quando semelhante, replicava essencialmente a saída do modelo de correlação. Em poucas exceções, houve uma pequena melhora entre 1% e 3%, considerada irrelevante para o problema.

Esses resultados sugerem que a forma como as regiões cerebrais se conectam, capturada pela matriz de correlação, é o aspecto mais importante para distinguir as classes.

### Notebooks do projeto

#### Matriz de conectividade

* **1.0\_correlation\_matrix.ipynb**
  Classifica utilizando somente 2 classes (Controles e Parkinson).

* **1.1\_3classes.ipynb**
  Classificação multiclasse (Controles, Prodomal, Parkinson).

* **1.2\_2etapas.ipynb**
  Adota a estratégia de classificação em duas etapas: primeiro Controles vs. Doentes (Prodomal + Parkinson), depois Prodomal vs. Parkinson, usando o mesmo vetor de conectividade.

* **1.3\_catch22.ipynb**
  Extraia características das séries BOLD com o Catch22 e depois extrai a matriz de conectividade dessas características que são
  vetorizadas e passadas a uma MLP.

* **1.4\_optuna.ipynb**
  Realiza otimização de hiperparâmetros do MLP usando Optuna.

#### Séries temporais

* **2.0\_time\_series.ipynb**
  Treina uma FCN 1D nas séries completas para classificação binária em 2 classes (Controle e Parkinson).

* **2.1\_3classes.ipynb**
  Ajusta o FCN para realizar classificação multiclasse (Controles, Prodomal, Parkinson).

* **2.2\_2etapas.ipynb**
  Executa o fluxo em duas etapas com FCN: Controle vs. Doente (Prodomal + Parkinson) e, depois, Prodomal vs. Parkinson.

* **2.3\_catch22.ipynb**
  Passa como entrada da FCN as características de cada uma das séries e não elas cruas a fim de verificar ganhos de desempenho.

#### Rede cerebral

* **3.0\_gnn.ipynb**
  Gera grafos (via OMST e thresholding) e treina Graph Neural Networks para classificação.

#### Multimodalidade

* **4.0\_multimodal.ipynb**
  Realiza o empilhamento (stacking) das previsões dos modelos baseados em séries temporais e matrizes de correlação para avaliar ganhos de performance com uma abordagem multimodal.

## Resultados

### Séries Temporais

|  Métrica | FCN (CNN 1D) | Catch22 + Classificador |
| :------: | :----------: | :---------------------: |
| Acurácia |    ------    |            —            |
| F1-Score |    ------    |            —            |
| Precisão |    ------    |            —            |
|  Recall  |    ------    |            —            |

### Matriz de Conectividade

|  Métrica |   MLP  |
| :------: | :----: |
| Acurácia | ------ |
| F1-Score | ------ |
| Precisão | ------ |
|  Recall  | ------ |

### Rede Cerebral (GNN)

|  Métrica |   GNN  |
| :------: | :----: |
| Acurácia | ------ |
| F1-Score | ------ |
| Precisão | ------ |
|  Recall  | ------ |
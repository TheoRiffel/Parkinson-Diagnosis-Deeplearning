# Dianóstico de Parkinson baseado em Redes Cerebrais Funcionais

Este repositório faz parte da entrega do trabalho da disciplina SCC0270 - Redes Neurais e Aprendizado Profundo (2025)

### Alunos:
    Gabriel da Costa Merlin - 12544420
    Théo Bruno Frey Riffel - 12547812
    
#TODO
    @frois explicar o trem do conda do pacote parkinson
    @todos botar os resultados de cada notebook na sessão Resultados

## Sobre o Projeto:

Este repositório contém um projeto para o diagnóstico da doença de Parkinson usando técnicas de aprendizagem profunda.

Os dados que usamos como base para o projeto são rs-fMRI (resting state functional Magnetic Ressonance Imaging). Os dados foram obtidos da base de dados PPMI (https://www.ppmi-info.org/PPMI). Os dados foram pré-processados com a ferramenta CONN toolbox (https://web.conn-toolbox.org/) a fim de remover ruídos de movimento físico dos pacientes durante o exame, ruídos provenientes da máquina e de efeitos adversos do campo magnético. **Os dados não estarão públicos por hora**, pois estão sendo realizados estudos e pesquisas sobre esses dados.

A ressonância magnética funcional (fMRI) possibilita medir o sinal dependente do nível de oxigênio no sangue (BOLD). Esse sinal serve como um indicador indireto da atividade neural, baseado no princípio do acoplamento neurovascular: o aumento do disparo neural em uma região do cérebro leva a um aumento localizado do fluxo sanguíneo e da oxigenação, o que, por sua vez, altera as propriedades magnéticas da hemoglobina e produz uma alteração detectável no sinal de Ressonância magnética funcional. 

Nos últimos anos, técnicas avançadas de neuroimagem se tornaram fundamentais na busca pela identificação de biomarcadores objetivos e não invasivos da patologia da DP. Entre elas, a ressonância magnética funcional em estado de repouso (rs-fMRI) surgiu como uma ferramenta particularmente poderosa. Em um ramo da neurociência chamdo Conectomas Cerebrais, onde procura-se entender a organização, funcionalidade e comportamento do cérebro humano, é comum a metodologia de utilizar dados rs-fMRI para identificar e caracterizar a organização cerebral de pacientes. 

Extraindo as séries BOLD de N regiões específicas do cérebro (comunmente chamadas de ROI's) e correlacionando-as com medidas de proximidade entre séries (Person, Mutual Information, DTW), montamos uma matriz A (NxN), onde cada elemento $ a_ij $ corresponde, ou tem a interpretação de, ao grau de interdependência funcional entre a região $i$ e a região $j$. Isso é, se o sinal BOLD da região $i$ é consistentemente similar ao sinal BOLD da região $j$, mostrado pelas métricas de correlação, dizemos que essas regiões compartilham funções parecidas no cérebro, ou que uma é dependente funcionalmente da outra. A partir dessa matriz A, chamada na literatura de **Matriz de Conectividade** é que extraímos informações valiosas sobre a organização e funcionamento do cérebro. Particularmente, essa matriz será essencial para diferenciar pacientes Controle saudáveis de pacientes com Parkinson.

Nesse repositório, abordamos vários métodos para o diagnóstico. Cada abordagem está listada em um notebook separado. A seguir, uma explicação breve de cada notebook e suas técnicas. Cada notebook está disponível no diretório 'notebooks'.

Contamos com dados de 66 pacientes saudáveis (Control), 153 pacientes com Parkinon Avançado e 188 pacientes com parkinson em estágio inicial.

Veremos no diretório Notebooks:

* **0_correlation_matrix.ipynb:** Modelo Baseline. Consiste em uma Shallow Network MLP (Multilayer Perceptron) básica.
* **2_pytorch.ipynb:** Rede Neural mais apurada. Contem normalizador Dropout, separação treino/validação/teste, a ajuste fino dos hiperparâmetros.
* **3_pure_timeseries.ipynb:** Consiste na aplicação de uma MLP diretamente nas séries temporais BOLD das ROI's.
* **4_multiclassifier.ipynb:** Modelo multiclassificador entre as 3 classes de pacientes. 
* **5_GNN.ipynb:** Aplicação de GNN (Graph Neural Network).

## Resultados





---

## Tecnologias Utilizadas

* **Linguagem:** Python 3
* **Redes Neurais:** Pytorch

---

## Configuração e Instalação

Siga estes passos para configurar e executar o projeto em sua máquina local.

### Pré-requisitos

* Python 3.8 ou superior
* `pip` (instalador de pacotes do Python)

### Passos para Instalação

1.  **Clonar o Repositório**
    ```bash
    git clone https://github.com/theoriffel/parkinson-diagnosis-deeplearning.git
    cd Parkinson-Diagnosis-Deeplearning
    ```

2.  **(Recomendado) Criar e Ativar um Ambiente Virtual**

    * No **macOS/Linux**:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

    * No **Windows**:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Instalar as Dependências**
    Instale os pacotes necessários usando o `pip`:
    ```bash
    pip install -r requirements.txt
    ```

---

## Como Usar

Com a configuração concluída, você pode executar os notebooks no diretório 'notebooks'.

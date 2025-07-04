import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

from tqdm import tqdm

class OMSTBuilder:
    """ Orthogonal Minimal Spanning Tree (OMST) Builder
    A classe encontra iterativamente Orthogonal Minimal Spanning Trees (OMSTs)
    e as adicionam a um grafo cumulativo, enquanto a
    Eficiência do Custo Global (GCE) aumentar.
    """

    def __init__(self, adj_matrix: np.ndarray):

        if not isinstance(adj_matrix, np.ndarray) or adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Input must be a square 2D NumPy array.")

        # Correspondente ao input 'CIJ' no script MATLAB.
        # Utilizando abs() para garantir pesos positivos, típico das matrizes de correlação.
        self.adj_matrix = np.abs(adj_matrix)
        self.n_nodes = self.adj_matrix.shape[0]

        # --- Inicializa as métricas do grafo original ---
        # Correspondente à 'cost_ini'
        self._initial_cost = np.sum(np.triu(self.adj_matrix))
        # Correspondente à 'E_ini'
        self._initial_ge = self._calculate_global_efficiency_from_matrix(self.adj_matrix)

        # --- Initialize graphs for the build process ---
        # Correspondente à 'CIJnotintree' - grafo das arestas restantes.
        # Utilizando grafos NetworkX para eficiência.
        self._residual_graph = self._create_distance_graph(self.adj_matrix)

        # --- Atributos públicos para armazenar os resultados ---
        self.omsts: List[nx.Graph] = []
        self.gce_scores: List[float] = []
        self.final_graph: nx.Graph = nx.Graph()

    def _create_distance_graph(self, matrix: np.ndarray) -> nx.Graph:
        """
        Auxiliar para criar um grafo ponderado por distância a partir de uma matriz de similaridade.
        Nos grafos de distância, menor peso é melhor.
        Correspondente à `1./CIJ` no script MATLAB.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            distance_matrix = 1 / matrix
        # Definido valores não-finitos (1/0 ou NaNs) para 0, indicando nenhum caminho.
        distance_matrix[~np.isfinite(distance_matrix)] = 0
        return nx.from_numpy_array(distance_matrix)

    def _calculate_global_efficiency_from_matrix(self, matrix: np.ndarray) -> float:
        """Calcula a eficiência global da matriz de similaridade."""
        dist_graph = self._create_distance_graph(matrix)
        return nx.global_efficiency(dist_graph)

    def _calculate_gce(self, graph: nx.Graph) -> float:
        """
        Calcula a Eficiência de Custo Global para o grafo cumulativo atual.
        Correspondente à formula `E/E_ini - cost(counter)`.
        """
        if self._initial_cost == 0 or self._initial_ge == 0:
            return -np.inf

        # Custo é a soma dos pesos (similaridade) como uma fração do custo total inicial.
        current_cost = graph.size(weight='weight') / self._initial_cost

        # Eficiência calculada no ggrafo de distância correspondente.
        current_ge = self._calculate_global_efficiency_from_matrix(nx.to_numpy_array(graph))
        
        return (current_ge / self._initial_ge) - current_cost

    def build(self, verbose: bool = True) -> nx.Graph:
        """
        Executa o processo de construção iterativo, correspondente ao loop principal
        `while delta > 0` no script MATLAB.

        Args:
            verbose: Se verdadeiro, imprime a pontuação do GCE em cada iteração.

        Retorna:
            O Grafo construído com OMST.
        """
        # Este grafo acumulará as arestas dos OMSTs encontrados.
        cumulative_graph = nx.Graph()
        previous_gce = -np.inf

        if verbose:
            print(f"Starting build. Initial GE={self._initial_ge:.4f}, Initial Cost={self._initial_cost:.2f}")
            print("-" * 30)

        for i in range(self.n_nodes * (self.n_nodes - 1) // 2): # Max possible iterations
            if self._residual_graph.number_of_edges() == 0:
                if verbose: print("\nNo more edges available. Stopping.")
                break

            # Encontrando o próximo MST a partir das arestas restantes.
            mst = nx.minimum_spanning_tree(self._residual_graph, weight='weight')
            if mst.number_of_edges() == 0:
                if verbose: print("\nGraph disconnected. Stopping.")
                break
            
            candidate_graph = cumulative_graph.copy()
            for u, v in mst.edges():
                weight = self.adj_matrix[u, v]
                candidate_graph.add_edge(u, v, weight=weight)

            # Calculando o GCE deste novo grafo candidato.
            current_gce = self._calculate_gce(candidate_graph)

            if verbose:
                print(f"Iteration {i+1}: GCE = {current_gce:.4f}")

            # Condição de parada: se GCE não melhorar, grafo ótimo encontrado.
            if current_gce < previous_gce:
                if verbose:
                    print(f"GCE decreased. Halting at {len(self.omsts)} OMST(s).")
                    print("-" * 30)
                break

            previous_gce = current_gce
            self.gce_scores.append(current_gce)
            self.omsts.append(mst)
            cumulative_graph = candidate_graph
 
            self._residual_graph.remove_edges_from(mst.edges())
        
        self.final_graph = cumulative_graph
        return self.final_graph

    def plot_gce_curve(self):
        """Traçando a pontuação do GCE em relação ao número de OMSTs adicionados."""
        if not self.gce_scores:
            print("No scores to plot. Run .build() first.")
            return

        plt.figure(figsize=(10, 6))
        iterations = range(1, len(self.gce_scores) + 1)
        plt.plot(iterations, self.gce_scores, marker='o', linestyle='-')
        
        max_gce = max(self.gce_scores)
        max_idx = self.gce_scores.index(max_gce)
        plt.plot(max_idx + 1, max_gce, 'r*', markersize=15, label=f'Max GCE: {max_gce:.4f}')
        
        plt.title("Global Cost Efficiency vs. Number of OMSTs")
        plt.xlabel("Number of OMSTs Added")
        plt.ylabel("Global Cost Efficiency (GCE)")
        plt.grid(True)
        plt.legend()
        plt.show()
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw
from tqdm import tqdm

# até onde eu sei essa função não tá sendo usada
def dtw_distance(time_series1: np.array, time_series2: np.array) -> float:
    distance, _ = fastdtw(time_series1.reshape(-1, 1), time_series2.reshape(-1, 1), dist=euclidean)
    return distance

# outra estimador para similaridade entre séries temporais (tipo um pearson)
def compute_dtw_matrix(time_series) -> np.array:
    """
    Calcula a matriz de distâncias DTW entre todos os pares de canais de uma série temporal.
    Retorna apenas os valores do triângulo superior (como na matriz de correlação).
    """
    # converte para array e coloca canais no eixo 0
    data = time_series.to_numpy().T  # shape (n_canais, n_amostras)
    # computa toda a matriz de distâncias
    dtw_net = compute_functional_network(data, dtw_distance)
    # extrai o triângulo superior (incluindo diagonal)
    iu = np.triu_indices(dtw_net.shape[0])
    return dtw_net[iu]


def compute_functional_network(time_series: np.array, distance_function: callable) -> np.array:
    """
    Calcula a matriz de distâncias entre canais de uma série temporal usando a função de distância fornecida.
    time_series.shape == (n_canais, n_amostras)
    """
    n = time_series.shape[0]
    functional_network = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i + 1, n):
            distance = distance_function(time_series[i].reshape(-1, 1), time_series[j].reshape(-1, 1))
            functional_network[i, j] = distance
            functional_network[j, i] = distance
    return functional_network


def compute_correlation_matrix(time_series: np.array, method: str = 'pearson') -> np.array:
    channels_number = time_series.shape[1]
    upper_triangular_indices = np.triu_indices(channels_number)
    correlation_matrix = time_series.corr(method=method).to_numpy()[upper_triangular_indices]
    return correlation_matrix
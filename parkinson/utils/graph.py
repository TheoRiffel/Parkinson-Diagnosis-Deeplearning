import numpy as np

from fastdtw import fastdtw
from tqdm import tqdm

def dtw_distance(time_series1: np.array, time_series2: np.array) -> float:
    distance, _ = fastdtw(time_series1.reshape(-1, 1), time_series2.reshape(-1, 1), dist=euclidean)
    return distance

def compute_functional_network(time_series: np.array, distance_function: callable) -> np.array:
    n = time_series.shape[0]
    functional_network = np.zeros((n, n))
    for i in tqdm(range(1, n), leave=True):
        for j in tqdm(range(i+1, n), leave=False):
            distance = distance_function(time_series[i].reshape(-1, 1), time_series[j].reshape(-1, 1))
            functional_network[i, j] = distance
            functional_network[j, i] = distance
    return functional_network

def compute_correlation_matrix(time_series: np.array) -> np.array:
    channels_number = time_series.shape[1]
    upper_triangular_indices = np.triu_indices(channels_number)
    correlation_matrix = time_series.corr().to_numpy()[upper_triangular_indices]
    return correlation_matrix
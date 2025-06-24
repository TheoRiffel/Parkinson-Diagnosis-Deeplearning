import numpy as np
import os
import sys
import time
from pathlib import Path
from joblib import Parallel, delayed
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from .utils import save_cache, load_cache, min_max_normalize

def dtw_distance(time_series1: np.array, time_series2: np.array) -> float:
    distance, _ = fastdtw(time_series1.reshape(-1, 1), time_series2.reshape(-1, 1), dist=euclidean)
    return distance

def compute_functional_network(time_series: np.array, distance_function: callable) -> np.array:
    n = time_series.shape[0]
    functional_network = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i + 1, n):
            distance = distance_function(time_series[i].reshape(-1, 1), time_series[j].reshape(-1, 1))
            functional_network[i, j] = distance
            functional_network[j, i] = distance
    return functional_network

def compute_dtw_matrix(time_series) -> np.array:
    data = time_series.to_numpy().T  # shape (n_canais, n_amostras)
    dtw_net = compute_functional_network(data, dtw_distance)
    iu = np.triu_indices(dtw_net.shape[0])
    return dtw_net[iu]

def compute_all_dtw_matrices(time_series_list, cache_path, n_jobs=4):
    if os.path.exists(cache_path):
        print(f"Carregando cache existente de {cache_path}...")
        completed_results = load_cache(cache_path)
    else:
        completed_results = {}

    total = len(time_series_list)
    indices_to_process = [i for i in range(total) if i not in completed_results]

    print(f"Total: {total} pacientes")
    print(f"Já computados: {len(completed_results)}")
    print(f"Restantes: {len(indices_to_process)}")

    def process_and_save(idx):
        print(f"Iniciando paciente {idx+1}...")
        start_time = time.time()
        ts = time_series_list[idx]
        result = compute_dtw_matrix(ts)
        elapsed = time.time() - start_time
        print(f"Paciente {idx+1} finalizado em {elapsed:.2f} segundos.")
        return idx, result

    for batch_start in range(0, len(indices_to_process), n_jobs):
        batch_indices = indices_to_process[batch_start:batch_start + n_jobs]
        print(f"\nProcessando batch {batch_start} a {batch_start + len(batch_indices) - 1}...")

        new_results = Parallel(n_jobs=n_jobs)(
            delayed(process_and_save)(i) for i in batch_indices
        )

        for idx, res in new_results:
            completed_results[idx] = res

        save_cache(cache_path, completed_results)
        print(f"Cache parcial salvo após batch {batch_start}.")

    ordered_results = [completed_results[i] for i in range(total)]
    return ordered_results

def dtw_correlation(time_series_list, group, n_jobs=6, **kwargs):
    root = Path(__file__).resolve().parents[3]
    sys.path.append('/')
    cache_dir = os.path.join(root, 'data/dtw_matrix')
    os.makedirs(cache_dir, exist_ok=True)
    if group == 'parkinson':
        cache_path = os.path.join(cache_dir, 'cache_dtw_parkinson_final.pkl')
    elif group == 'control':
        cache_path = os.path.join(cache_dir, 'cache_dtw_control_final.pkl')
    else:
        raise ValueError("group deve ser 'parkinson' ou 'control'")
    dtw_matrices = compute_all_dtw_matrices(time_series_list, cache_path, n_jobs=n_jobs)
    return [min_max_normalize(mat) for mat in dtw_matrices]